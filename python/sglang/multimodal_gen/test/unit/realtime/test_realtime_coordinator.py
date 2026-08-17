# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

import pytest

from sglang.multimodal_gen.runtime.realtime.coordinator import (
    CoordinatorRejected,
    DynamoDBCoordinatorStore,
    HTTPWorkerReservationClient,
    InMemoryCoordinatorStore,
    RealtimeCoordinator,
    SessionAssignment,
    WorkerHeartbeat,
    WorkerSlot,
)


def _heartbeat(
    worker_id: str,
    role: str,
    *,
    capacity: int = 1,
    az: str = "us-east-2a",
    model_revision: str = "minwm-r1",
    vae_fingerprint: str = "taew2_2",
    worker_epoch: str = "epoch-a",
    lifecycle: str = "ready",
    active_sessions: int = 0,
    queue_depth: int = 0,
    service_time_ms: float = 0,
    drain_deadline: float | None = None,
):
    return WorkerHeartbeat(
        worker_id=worker_id,
        role=role,
        endpoint=f"ws://{worker_id}.cluster.local/generate",
        az=az,
        capacity=capacity,
        model_revision=model_revision,
        vae_fingerprint=vae_fingerprint,
        worker_epoch=worker_epoch,
        lifecycle=lifecycle,
        active_sessions=active_sessions,
        runnable_sessions=active_sessions,
        blocked_sessions=0,
        queue_depth=queue_depth,
        service_time_ms=service_time_ms,
        reservation_endpoint=f"http://{worker_id}.cluster.local/v1/realtime_worker",
        drain_deadline=drain_deadline,
    )


def _create_dynamodb_coordinator_table(client, table_name):
    client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
            {"AttributeName": "allocation_key", "AttributeType": "S"},
            {"AttributeName": "allocation_sort", "AttributeType": "S"},
        ],
        GlobalSecondaryIndexes=[
            {
                "IndexName": "allocation-index",
                "KeySchema": [
                    {"AttributeName": "allocation_key", "KeyType": "HASH"},
                    {"AttributeName": "allocation_sort", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            }
        ],
        BillingMode="PAY_PER_REQUEST",
    )


def test_coordinator_atomically_pairs_compatible_worker_slots():
    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        coordinator = RealtimeCoordinator(store, wait_timeout_s=0)
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser", capacity=2))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae", capacity=2))

        assignment = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        assert assignment.denoiser.worker_id == "denoiser-a"
        assert assignment.vae.worker_id == "vae-a"
        assert assignment.denoiser.slot_index == 0
        assert assignment.vae.slot_index == 0
        assert assignment.token
        return coordinator, assignment

    coordinator, assignment = asyncio.run(run())
    assert coordinator is not None
    assert assignment.session_id == "session-a"


def test_capacity_snapshot_combines_waiting_load_free_slots_and_drain_state():
    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        coordinator = RealtimeCoordinator(store, wait_timeout_s=0)
        await coordinator.heartbeat(
            _heartbeat(
                "denoiser-ready",
                "denoiser",
                capacity=4,
                active_sessions=3,
                queue_depth=2,
                service_time_ms=450,
            )
        )
        await coordinator.heartbeat(
            _heartbeat(
                "denoiser-draining",
                "denoiser",
                capacity=4,
                lifecycle="draining",
                active_sessions=1,
            )
        )
        await coordinator.heartbeat(
            _heartbeat(
                "vae-ready",
                "vae",
                capacity=16,
                active_sessions=4,
                queue_depth=1,
                service_time_ms=30,
            )
        )
        await store.waiting_started("waiter-a")

        snapshot = await coordinator.capacity_snapshot()
        await store.waiting_finished("waiter-a")
        after = await coordinator.capacity_snapshot()

        assert snapshot["roles"]["denoiser"] == {
            "waiting_sessions": 1,
            "active_sessions": 4,
            "queued_sessions": 2,
            "free_slots": 1,
            "draining_workers": 1,
        }
        assert snapshot["roles"]["vae"] == {
            "waiting_sessions": 1,
            "active_sessions": 4,
            "queued_sessions": 1,
            "free_slots": 12,
            "draining_workers": 0,
        }
        assert after["roles"]["denoiser"]["waiting_sessions"] == 0

    asyncio.run(run())


def test_dynamodb_capacity_snapshot_uses_shared_ttl_demand_records():
    class FakeClient:
        def __init__(self):
            self.puts = []
            self.deletes = []
            self.queries = []

        def put_item(self, **kwargs):
            self.puts.append(kwargs)

        def delete_item(self, **kwargs):
            self.deletes.append(kwargs)

        def query(self, **kwargs):
            self.queries.append(kwargs)
            role = kwargs["ExpressionAttributeValues"][":allocation"]["S"].split(
                "#", 1
            )[1]
            capacity = 4 if role == "denoiser" else 16
            active = 3 if role == "denoiser" else 4
            return {
                "Items": [
                    {
                        "item_type": {"S": "worker"},
                        "role": {"S": role},
                        "worker_id": {"S": f"{role}-a"},
                        "lifecycle": {"S": "ready"},
                        "capacity": {"N": str(capacity)},
                        "active_sessions": {"N": str(active)},
                        "queue_depth": {"N": "1"},
                        "heartbeat_expires_at": {"N": "200"},
                    },
                    {
                        "item_type": {"S": "capacity_demand"},
                        "demand_expires_at": {"N": "160"},
                    },
                ]
            }

    async def run():
        client = FakeClient()
        store = DynamoDBCoordinatorStore(
            "minwm-realtime-coordinator",
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 100,
            client=client,
        )
        await store.waiting_started("waiter-a")
        snapshot = await store.capacity_snapshot()
        await store.waiting_finished("waiter-a")

        assert len(client.puts) == 2
        assert {item["Item"]["allocation_key"]["S"] for item in client.puts} == {
            "CAPACITY#denoiser",
            "CAPACITY#vae",
        }
        assert len(client.queries) == 2
        assert snapshot["roles"]["denoiser"]["waiting_sessions"] == 1
        assert snapshot["roles"]["denoiser"]["free_slots"] == 1
        assert snapshot["roles"]["vae"]["free_slots"] == 12
        assert len(client.deletes) == 2

    asyncio.run(run())


def test_coordinator_rejects_second_session_for_the_same_user():
    async def run():
        coordinator = RealtimeCoordinator(
            InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30),
            wait_timeout_s=0,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser", capacity=2))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae", capacity=2))
        await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        with pytest.raises(CoordinatorRejected, match="USER_SESSION_LIMIT"):
            await coordinator.admit(
                user_id="user-a",
                session_id="session-b",
                generation_id="generation-b",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )

    asyncio.run(run())


def test_coordinator_does_not_leak_a_partial_worker_reservation():
    async def run():
        coordinator = RealtimeCoordinator(
            InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30),
            wait_timeout_s=0,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coordinator.admit(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )

        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        assignment = await coordinator.admit(
            user_id="user-b",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert assignment.denoiser.slot_index == 0

    asyncio.run(run())


def test_coordinator_prefers_same_az_and_filters_incompatible_workers():
    async def run():
        coordinator = RealtimeCoordinator(
            InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30),
            wait_timeout_s=0,
        )
        await coordinator.heartbeat(
            _heartbeat("denoiser-a", "denoiser", az="us-east-2a")
        )
        await coordinator.heartbeat(
            _heartbeat("vae-wrong", "vae", az="us-east-2a", vae_fingerprint="wrong")
        )
        await coordinator.heartbeat(_heartbeat("vae-cross-az", "vae", az="us-east-2b"))
        await coordinator.heartbeat(_heartbeat("vae-same-az", "vae", az="us-east-2a"))

        assignment = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert assignment.vae.worker_id == "vae-same-az"

    asyncio.run(run())


def test_coordinator_excludes_draining_workers_from_new_allocations():
    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        await store.heartbeat(
            _heartbeat("denoiser-draining", "denoiser", lifecycle="draining")
        )
        await store.heartbeat(_heartbeat("denoiser-ready", "denoiser"))
        await store.heartbeat(_heartbeat("vae-ready", "vae"))

        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        assert assignment.denoiser.worker_id == "denoiser-ready"

    asyncio.run(run())


def test_coordinator_routes_to_lower_normalized_load_queue_and_service_time():
    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        await store.heartbeat(
            _heartbeat(
                "denoiser-loaded",
                "denoiser",
                capacity=4,
                active_sessions=3,
                queue_depth=2,
                service_time_ms=20,
            )
        )
        await store.heartbeat(
            _heartbeat(
                "denoiser-light",
                "denoiser",
                capacity=4,
                active_sessions=1,
                queue_depth=0,
                service_time_ms=5,
            )
        )
        await store.heartbeat(_heartbeat("vae-a", "vae", capacity=4))

        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        assert assignment.denoiser.worker_id == "denoiser-light"

    asyncio.run(run())


def test_coordinator_waiting_admission_wakes_when_assignment_is_released():
    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        coordinator = RealtimeCoordinator(store, wait_timeout_s=1)
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        first = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        waiting = asyncio.create_task(
            coordinator.admit(
                user_id="user-b",
                session_id="session-b",
                generation_id="generation-b",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )
        )
        await asyncio.sleep(0.01)
        assert not waiting.done()
        assert (await coordinator.capacity_snapshot())["roles"]["denoiser"][
            "waiting_sessions"
        ] == 1

        await coordinator.release(first)
        second = await asyncio.wait_for(waiting, timeout=0.5)
        assert second.session_id == "session-b"
        assert (await coordinator.capacity_snapshot())["roles"]["denoiser"][
            "waiting_sessions"
        ] == 0

    asyncio.run(run())


def test_coordinator_renew_fences_worker_restart_and_heartbeat_loss():
    async def run():
        now = [100.0]
        store = InMemoryCoordinatorStore(
            ttl_s=60,
            worker_ttl_s=5,
            clock=lambda: now[0],
        )
        await store.heartbeat(
            _heartbeat("denoiser-a", "denoiser", worker_epoch="epoch-old")
        )
        await store.heartbeat(_heartbeat("vae-a", "vae"))
        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        await store.heartbeat(
            _heartbeat("denoiser-a", "denoiser", worker_epoch="epoch-new")
        )
        with pytest.raises(CoordinatorRejected, match="WORKER_LOST"):
            await store.renew(assignment)

        await store.heartbeat(
            _heartbeat("denoiser-a", "denoiser", worker_epoch="epoch-old")
        )
        now[0] = 106.0
        with pytest.raises(CoordinatorRejected, match="WORKER_LOST"):
            await store.renew(assignment)

    asyncio.run(run())


def test_coordinator_allows_draining_worker_renew_only_before_deadline():
    async def run():
        now = [100.0]
        wall_now = [1_000.0]
        store = InMemoryCoordinatorStore(
            ttl_s=60,
            worker_ttl_s=30,
            clock=lambda: now[0],
            wall_clock=lambda: wall_now[0],
        )
        await store.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await store.heartbeat(_heartbeat("vae-a", "vae"))
        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        await store.heartbeat(
            _heartbeat(
                "denoiser-a",
                "denoiser",
                lifecycle="draining",
                drain_deadline=1_010.0,
            )
        )

        assignment = await store.renew(assignment)
        wall_now[0] = 1_011.0
        with pytest.raises(CoordinatorRejected, match="WORKER_LOST"):
            await store.renew(assignment)

    asyncio.run(run())


def test_coordinator_partial_worker_reserve_rolls_back_and_retries_another_pair():
    class ReservationClient:
        def __init__(self):
            self.reserved = []
            self.released = []

        async def reserve(self, slot, **identity):
            self.reserved.append((slot.worker_id, identity["token"]))
            if slot.worker_id == "vae-bad":
                raise RuntimeError("worker rejected reservation")

        async def release(self, slot, *, token):
            self.released.append((slot.worker_id, token))

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        reservations = ReservationClient()
        coordinator = RealtimeCoordinator(
            store,
            wait_timeout_s=1,
            reservation_client=reservations,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-bad", "vae", service_time_ms=0))
        await coordinator.heartbeat(_heartbeat("vae-good", "vae", service_time_ms=1))

        assignment = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        assert assignment.vae.worker_id == "vae-good"
        failed_token = reservations.reserved[0][1]
        assert ("denoiser-a", failed_token) in reservations.released
        assert ("vae-bad", failed_token) in reservations.released
        assert assignment.token != failed_token

    asyncio.run(run())


def test_coordinator_expires_stale_workers_and_reclaims_expired_assignments():
    async def run():
        now = [100.0]
        store = InMemoryCoordinatorStore(
            ttl_s=5,
            worker_ttl_s=3,
            clock=lambda: now[0],
        )
        coordinator = RealtimeCoordinator(store, wait_timeout_s=0)
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        first = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        now[0] = 104.0
        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coordinator.admit(
                user_id="user-b",
                session_id="session-b",
                generation_id="generation-b",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )

        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        now[0] = 106.0
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        second = await coordinator.admit(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert second.token != first.token

    asyncio.run(run())


def test_coordinator_renew_and_release_are_fenced_and_idempotent():
    async def run():
        coordinator = RealtimeCoordinator(
            InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30),
            wait_timeout_s=0,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        assignment = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        renewed = await coordinator.renew(assignment)
        assert renewed.token == assignment.token
        assert renewed.expires_at >= assignment.expires_at

        await coordinator.release(renewed)
        await coordinator.release(renewed)
        replacement = await coordinator.admit(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"

    asyncio.run(run())


def test_worker_reservation_http_release_treats_consumed_slot_as_idempotent():
    class Response:
        status_code = 409

        @staticmethod
        def json():
            return {"detail": {"reason": "RESERVATION_OWNER_MISMATCH"}}

        @staticmethod
        def raise_for_status():
            raise AssertionError("idempotent release should not raise")

    class Client:
        async def delete(self, url):
            assert url == "http://worker/v1/realtime_worker/reservations/token-a"
            return Response()

    async def run():
        client = HTTPWorkerReservationClient(client=Client())
        await client.release(
            WorkerSlot(
                worker_id="denoiser-a",
                role="denoiser",
                endpoint="ws://denoiser/generate",
                az="us-east-2a",
                slot_index=0,
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
                worker_epoch="epoch-a",
                reservation_endpoint="http://worker/v1/realtime_worker",
            ),
            token="token-a",
        )

    asyncio.run(run())


def test_coordinator_release_frees_store_when_worker_cleanup_is_best_effort():
    class ReservationClient:
        def __init__(self):
            self.release_attempts = 0

        async def reserve(self, slot, **_identity):
            del slot

        async def release(self, slot, *, token):
            del slot, token
            self.release_attempts += 1
            raise RuntimeError("worker cleanup already owns the reservation")

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        reservations = ReservationClient()
        coordinator = RealtimeCoordinator(
            store,
            wait_timeout_s=0,
            reservation_client=reservations,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))
        assignment = await coordinator.admit(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        await coordinator.release(assignment)

        assert reservations.release_attempts == 6
        replacement = await store.acquire(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"

    asyncio.run(run())


def test_dynamodb_coordinator_admission_fences_slots_with_worker_heartbeats():
    class TransactionCanceledException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.transactions = []

        def query(self, **kwargs):
            allocation_key = kwargs["ExpressionAttributeValues"][":allocation"]["S"]
            if allocation_key.startswith("DENOISER#"):
                role = "denoiser"
                worker_id = "denoiser-a"
                endpoint = "ws://denoiser-a/generate"
            else:
                role = "vae"
                worker_id = "vae-a"
                endpoint = "ws://vae-a/decode"
            return {
                "Items": [
                    {
                        "pk": {"S": f"SLOT#{role}#{worker_id}#0000"},
                        "sk": {"S": "LEASE"},
                        "role": {"S": role},
                        "worker_id": {"S": worker_id},
                        "endpoint": {"S": endpoint},
                        "az": {"S": "us-east-2a"},
                        "slot_index": {"N": "0"},
                        "model_revision": {"S": "minwm-r1"},
                        "vae_fingerprint": {"S": "taew2_2"},
                        "heartbeat_expires_at": {"N": "9999999999"},
                    }
                ]
            }

        def transact_write_items(self, *, TransactItems):
            self.transactions.append(TransactItems)

    async def run():
        client = FakeClient()
        store = DynamoDBCoordinatorStore(
            "minwm-realtime-coordinator",
            ttl_s=60,
            worker_ttl_s=30,
            client=client,
        )
        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert assignment.denoiser.worker_id == "denoiser-a"
        assert assignment.vae.worker_id == "vae-a"
        assert len(client.transactions) == 1
        transaction = client.transactions[0]
        assert len(transaction) == 6
        keys = {
            (
                item["Put"]["Item"]["pk"]["S"]
                if "Put" in item
                else (
                    item["Update"]["Key"]["pk"]["S"]
                    if "Update" in item
                    else item["ConditionCheck"]["Key"]["pk"]["S"]
                )
            )
            for item in transaction
        }
        assert keys == {
            "USER#user-a",
            "SESSION#session-a",
            "SLOT#denoiser#denoiser-a#0000",
            "SLOT#vae#vae-a#0000",
            "WORKER#denoiser-a",
            "WORKER#vae-a",
        }
        checks = [
            item["ConditionCheck"] for item in transaction if "ConditionCheck" in item
        ]
        assert all(
            "#capacity > :slot_index" in check["ConditionExpression"]
            and "worker_epoch = :worker_epoch" in check["ConditionExpression"]
            and "heartbeat_expires_at > :now" in check["ConditionExpression"]
            and "#lifecycle = :ready" in check["ConditionExpression"]
            for check in checks
        )

    asyncio.run(run())


def test_dynamodb_slot_query_paginates_past_filtered_stale_slots():
    class FakeClient:
        def __init__(self):
            self.queries = []

        def query(self, **kwargs):
            self.queries.append(kwargs)
            if "ExclusiveStartKey" not in kwargs:
                return {
                    "Items": [],
                    "LastEvaluatedKey": {
                        "allocation_key": {"S": "DENOISER#minwm-r1"},
                        "allocation_sort": {"S": "stale-worker#0001"},
                    },
                }
            return {
                "Items": [
                    {
                        "pk": {"S": "SLOT#denoiser#denoiser-live#0000"},
                        "sk": {"S": "LEASE"},
                        "role": {"S": "denoiser"},
                        "worker_id": {"S": "denoiser-live"},
                        "endpoint": {"S": "ws://denoiser-live/generate"},
                        "az": {"S": "us-east-2a"},
                        "slot_index": {"N": "0"},
                        "model_revision": {"S": "minwm-r1"},
                        "vae_fingerprint": {"S": "taew2_2"},
                        "heartbeat_expires_at": {"N": "9999999999"},
                    }
                ]
            }

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
        candidate_limit=2,
    )

    slots = store._query_slots_sync(
        "denoiser",
        model_revision="minwm-r1",
        vae_fingerprint="taew2_2",
        now_epoch=100,
    )

    assert [slot.worker_id for slot in slots] == ["denoiser-live"]
    assert len(client.queries) == 2
    assert client.queries[1]["ExclusiveStartKey"] == {
        "allocation_key": {"S": "DENOISER#minwm-r1"},
        "allocation_sort": {"S": "stale-worker#0001"},
    }


def test_dynamodb_candidate_pairing_spreads_each_burst_in_stable_order():
    denoisers = [
        WorkerSlot(
            worker_id=f"denoiser-{index}",
            role="denoiser",
            endpoint=f"ws://denoiser-{index}/generate",
            az="us-east-2a",
            slot_index=0,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        for index in range(8)
    ]
    vaes = [
        WorkerSlot(
            worker_id=f"vae-{index}",
            role="vae",
            endpoint=f"ws://vae-{index}/decode",
            az="us-east-2a",
            slot_index=0,
            model_revision="all",
            vae_fingerprint="taew2_2",
        )
        for index in range(8)
    ]

    first_denoisers = set()
    for index in range(32):
        pairs = DynamoDBCoordinatorStore._candidate_pairs(
            denoisers,
            vaes,
            identity=f"user-{index}:session-{index}:generation-{index}",
        )
        assert len(pairs) == 8
        assert len({pair[0].worker_id for pair in pairs}) == 8
        assert len({pair[1].worker_id for pair in pairs}) == 8
        first_denoisers.add(pairs[0][0].worker_id)

    assert len(first_denoisers) == 1


def test_dynamodb_candidate_pairing_prefers_worker_with_more_free_slots():
    def slots(worker_id: str, free_slots: int) -> list[WorkerSlot]:
        return [
            WorkerSlot(
                worker_id=worker_id,
                role="denoiser",
                endpoint=f"ws://{worker_id}/generate",
                az="us-east-2a",
                slot_index=index,
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
                active_sessions=0,
                capacity=4,
            )
            for index in range(free_slots)
        ]

    denoisers = slots("denoiser-mostly-busy", 1) + slots("denoiser-idle", 4)
    vaes = [
        WorkerSlot(
            worker_id="vae-a",
            role="vae",
            endpoint="ws://vae-a/decode",
            az="us-east-2a",
            slot_index=0,
            model_revision="all",
            vae_fingerprint="taew2_2",
        )
    ]

    pairs = DynamoDBCoordinatorStore._candidate_pairs(
        denoisers,
        vaes,
        identity="user-1:session-1:generation-1",
    )

    assert pairs[0][0].worker_id == "denoiser-idle"


def test_dynamodb_candidate_pairing_exhausts_worker_layer_before_next_slot():
    denoisers = [
        WorkerSlot(
            worker_id=f"denoiser-{worker_index}",
            role="denoiser",
            endpoint=f"ws://denoiser-{worker_index}/generate",
            az="us-east-2a",
            slot_index=slot_index,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            capacity=4,
        )
        for worker_index in range(8)
        for slot_index in range(4)
    ]
    vaes = [
        WorkerSlot(
            worker_id="vae-a",
            role="vae",
            endpoint="ws://vae-a/decode",
            az="us-east-2a",
            slot_index=slot_index,
            model_revision="all",
            vae_fingerprint="taew2_2",
            capacity=16,
        )
        for slot_index in range(16)
    ]

    pairs = DynamoDBCoordinatorStore._candidate_pairs(
        denoisers,
        vaes,
        identity="burst-a",
    )

    first_layer = [pair[0] for pair in pairs[:8]]
    assert {slot.worker_id for slot in first_layer} == {
        f"denoiser-{index}" for index in range(8)
    }
    assert {slot.slot_index for slot in first_layer} == {0}

    competing_pairs = DynamoDBCoordinatorStore._candidate_pairs(
        denoisers,
        vaes,
        identity="burst-b",
    )
    assert [
        (pair[0].worker_id, pair[0].slot_index, pair[1].slot_index)
        for pair in competing_pairs
    ] == [(pair[0].worker_id, pair[0].slot_index, pair[1].slot_index) for pair in pairs]


def test_dynamodb_admission_requeries_after_a_stale_candidate_snapshot():
    class TransactionCanceledException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.query_counts = {"denoiser": 0, "vae": 0}
            self.transactions = 0

        @staticmethod
        def _slot(role, index):
            worker_id = f"{role}-{index}"
            return {
                "pk": {"S": f"SLOT#{role}#{worker_id}#{index:04d}"},
                "sk": {"S": "LEASE"},
                "role": {"S": role},
                "worker_id": {"S": worker_id},
                "endpoint": {"S": f"ws://{worker_id}/generate"},
                "az": {"S": "us-east-2a"},
                "slot_index": {"N": str(index)},
                "model_revision": {"S": "minwm-r1"},
                "vae_fingerprint": {"S": "taew2_2"},
                "heartbeat_expires_at": {"N": "9999999999"},
            }

        def query(self, **kwargs):
            allocation_key = kwargs["ExpressionAttributeValues"][":allocation"]["S"]
            role = "denoiser" if allocation_key.startswith("DENOISER#") else "vae"
            self.query_counts[role] += 1
            indices = [0, 1] if self.query_counts[role] == 1 else [1]
            return {"Items": [self._slot(role, index) for index in indices]}

        def transact_write_items(self, *, TransactItems):
            self.transactions += 1
            if self.transactions <= 2:
                raise TransactionCanceledException("candidate was leased concurrently")

    async def run():
        client = FakeClient()
        store = DynamoDBCoordinatorStore(
            "minwm-realtime-coordinator",
            ttl_s=60,
            worker_ttl_s=30,
            client=client,
        )
        assignment = await store.acquire(
            user_id="user-a",
            session_id="session-a",
            generation_id="generation-a",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )

        assert assignment.denoiser.worker_id == "denoiser-1"
        assert assignment.vae.worker_id == "vae-1"
        assert client.query_counts == {"denoiser": 2, "vae": 2}
        assert client.transactions == 3

    asyncio.run(run())


def test_dynamodb_heartbeat_retries_a_transient_transaction_conflict():
    class TransactionConflictException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionConflictException = TransactionConflictException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.slot_updates = 0

        def put_item(self, **kwargs):
            return None

        def transact_write_items(self, *, TransactItems):
            self.slot_updates += 1
            if self.slot_updates == 1:
                raise TransactionConflictException("transaction in progress")
            assert any("ConditionCheck" in item for item in TransactItems)
            assert any("Update" in item for item in TransactItems)

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
    )

    store._heartbeat_sync(_heartbeat("denoiser-a", "denoiser"))

    assert client.slot_updates == 2


def test_dynamodb_heartbeat_clamps_advertised_denoiser_capacity():
    class TransactionConflictException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionConflictException = TransactionConflictException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.worker_item = None
            self.slot_updates = []

        def put_item(self, *, Item, **_kwargs):
            self.worker_item = Item

        def transact_write_items(self, *, TransactItems):
            self.slot_updates.extend(
                item["Update"] for item in TransactItems if "Update" in item
            )

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
        capacity_limits={"denoiser": 1},
    )

    store._heartbeat_sync(_heartbeat("denoiser-a", "denoiser", capacity=4))

    assert client.worker_item["capacity"] == {"N": "1"}
    assert len(client.slot_updates) == 1
    assert client.slot_updates[0]["ExpressionAttributeValues"][":capacity"] == {
        "N": "1"
    }


def test_dynamodb_heartbeat_allows_four_sessions_per_denoiser_gpu():
    class TransactionConflictException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionConflictException = TransactionConflictException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.worker_item = None
            self.slot_updates = []

        def put_item(self, *, Item, **_kwargs):
            self.worker_item = Item

        def transact_write_items(self, *, TransactItems):
            self.slot_updates.extend(
                item["Update"] for item in TransactItems if "Update" in item
            )

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
        capacity_limits={"denoiser": 4},
    )

    store._heartbeat_sync(_heartbeat("denoiser-a", "denoiser", capacity=4))

    assert client.worker_item["capacity"] == {"N": "4"}
    assert len(client.slot_updates) == 4
    assert {
        update["ExpressionAttributeValues"][":slot_index"]["N"]
        for update in client.slot_updates
    } == {"0", "1", "2", "3"}


def test_dynamodb_heartbeat_persists_epoch_lifecycle_and_worker_load():
    class TransactionConflictException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionConflictException = TransactionConflictException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.worker_item = None
            self.slot_update = None

        def put_item(self, **kwargs):
            self.worker_item = kwargs["Item"]

        def transact_write_items(self, *, TransactItems):
            self.slot_update = next(
                item["Update"] for item in TransactItems if "Update" in item
            )

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
    )
    heartbeat = _heartbeat(
        "denoiser-a",
        "denoiser",
        active_sessions=2,
        queue_depth=3,
        service_time_ms=12.5,
    )

    store._heartbeat_sync(heartbeat)

    assert client.worker_item["worker_epoch"] == {"S": "epoch-a"}
    assert client.worker_item["lifecycle"] == {"S": "ready"}
    assert client.worker_item["active_sessions"] == {"N": "2"}
    values = client.slot_update["ExpressionAttributeValues"]
    names = client.slot_update["ExpressionAttributeNames"]
    assert names["#capacity"] == "capacity"
    assert "#capacity = :capacity" in client.slot_update["UpdateExpression"]
    assert values[":worker_epoch"] == {"S": "epoch-a"}
    assert values[":heartbeat_generation"] == client.worker_item["heartbeat_generation"]
    assert values[":queue_depth"] == {"N": "3"}
    assert values[":service_time_ms"] == {"N": "12.5"}


def test_dynamodb_capacity_shrink_atomically_retires_slots_and_grows():
    class TransactionCanceledException(Exception):
        pass

    class TransactionConflictException(Exception):
        pass

    class ConditionalCheckFailedException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException
    FakeExceptions.TransactionConflictException = TransactionConflictException
    FakeExceptions.ConditionalCheckFailedException = ConditionalCheckFailedException

    def slot_item(role, worker_id, slot_index, capacity):
        model_revision = "minwm-r1" if role == "denoiser" else "all"
        return {
            "pk": {"S": f"SLOT#{role}#{worker_id}#{slot_index:04d}"},
            "sk": {"S": "LEASE"},
            "item_type": {"S": "worker_slot"},
            "role": {"S": role},
            "worker_id": {"S": worker_id},
            "endpoint": {"S": f"ws://{worker_id}/generate"},
            "az": {"S": "us-east-2a"},
            "slot_index": {"N": str(slot_index)},
            "model_revision": {"S": model_revision},
            "vae_fingerprint": {"S": "taew2_2"},
            "worker_epoch": {"S": "epoch-a"},
            "lifecycle": {"S": "ready"},
            "capacity": {"N": str(capacity)},
            "active_sessions": {"N": "0"},
            "runnable_sessions": {"N": "0"},
            "blocked_sessions": {"N": "0"},
            "queue_depth": {"N": "0"},
            "service_time_ms": {"N": "0"},
            "reservation_endpoint": {"S": f"http://{worker_id}/worker"},
            "heartbeat_expires_at": {"N": "200"},
            "allocation_key": {
                "S": ("DENOISER#minwm-r1" if role == "denoiser" else "VAE#taew2_2")
            },
            "allocation_sort": {"S": f"us-east-2a#{worker_id}#{slot_index:04d}"},
            "ttl": {"N": "86600"},
        }

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self, *, retire_failures=1):
            self.retire_failures = retire_failures
            self.retire_attempts = 0
            self.put_calls = 0
            self.order = []
            self.acquire_transaction = None
            self.workers = {
                "denoiser-a": {
                    "pk": {"S": "WORKER#denoiser-a"},
                    "sk": {"S": "HEARTBEAT"},
                    "role": {"S": "denoiser"},
                    "capacity": {"N": "2"},
                    "worker_epoch": {"S": "epoch-a"},
                    "lifecycle": {"S": "ready"},
                    "heartbeat_expires_at": {"N": "200"},
                },
                "vae-a": {
                    "pk": {"S": "WORKER#vae-a"},
                    "sk": {"S": "HEARTBEAT"},
                    "role": {"S": "vae"},
                    "capacity": {"N": "1"},
                    "worker_epoch": {"S": "epoch-a"},
                    "lifecycle": {"S": "ready"},
                    "heartbeat_expires_at": {"N": "200"},
                },
            }
            self.slots = {
                "SLOT#denoiser#denoiser-a#0000": slot_item(
                    "denoiser", "denoiser-a", 0, 2
                ),
                "SLOT#denoiser#denoiser-a#0001": slot_item(
                    "denoiser", "denoiser-a", 1, 2
                ),
                "SLOT#vae#vae-a#0000": slot_item("vae", "vae-a", 0, 1),
            }
            for pk in (
                "SLOT#denoiser#denoiser-a#0001",
                "SLOT#vae#vae-a#0000",
            ):
                self.slots[pk].update(
                    {
                        "lease_token": {"S": "active-token"},
                        "user_id": {"S": "user-a"},
                        "session_id": {"S": "session-a"},
                        "generation_id": {"S": "generation-a"},
                        "lease_expires_at": {"N": "180"},
                    }
                )

        def get_item(self, *, Key, **_kwargs):
            pk = Key["pk"]["S"]
            if pk.startswith("WORKER#"):
                return {"Item": self.workers.get(pk.removeprefix("WORKER#"))}
            if pk.startswith("SLOT#"):
                return {"Item": self.slots.get(pk)}
            return {}

        def put_item(
            self,
            *,
            Item,
            ConditionExpression=None,
            ExpressionAttributeValues=None,
            **_kwargs,
        ):
            self.put_calls += 1
            worker_id = Item["pk"]["S"].removeprefix("WORKER#")
            current = self.workers.get(worker_id)
            if ConditionExpression == "attribute_not_exists(pk)" and current:
                raise ConditionalCheckFailedException("worker already exists")
            if (
                ConditionExpression
                and ConditionExpression != "attribute_not_exists(pk)"
            ):
                values = ExpressionAttributeValues
                if (
                    current is None
                    or current["role"] != values[":previous_role"]
                    or current["capacity"] != values[":previous_capacity"]
                    or (
                        ":previous_worker_epoch" in values
                        and current.get("worker_epoch")
                        != values[":previous_worker_epoch"]
                    )
                    or (
                        ":previous_heartbeat_generation" in values
                        and current.get("heartbeat_generation")
                        != values[":previous_heartbeat_generation"]
                    )
                    or (
                        ":previous_heartbeat_generation" not in values
                        and "heartbeat_generation" in current
                    )
                ):
                    raise ConditionalCheckFailedException("stale worker snapshot")
            self.workers[worker_id] = dict(Item)
            self.order.append(f"put:{Item['capacity']['N']}")

        def update_item(
            self,
            *,
            Key,
            ExpressionAttributeValues,
            ConditionExpression=None,
            **_kwargs,
        ):
            pk = Key["pk"]["S"]
            item = self.slots.setdefault(pk, {"pk": Key["pk"], "sk": Key["sk"]})
            values = ExpressionAttributeValues
            if ConditionExpression and "previous_worker_epoch" in ConditionExpression:
                current_epoch = item.get("worker_epoch")
                current_capacity = item.get("capacity")
                target = (values[":worker_epoch"], values[":capacity"])
                if (
                    current_epoch is not None
                    and current_epoch != values[":previous_worker_epoch"]
                    and (current_epoch, current_capacity) != target
                ):
                    raise ConditionalCheckFailedException("stale slot snapshot")
            field_values = {
                "item_type": ":item_type",
                "role": ":role",
                "worker_id": ":worker_id",
                "endpoint": ":endpoint",
                "az": ":az",
                "slot_index": ":slot_index",
                "model_revision": ":model_revision",
                "vae_fingerprint": ":vae_fingerprint",
                "worker_epoch": ":worker_epoch",
                "heartbeat_generation": ":heartbeat_generation",
                "lifecycle": ":lifecycle",
                "capacity": ":capacity",
                "active_sessions": ":active_sessions",
                "runnable_sessions": ":runnable_sessions",
                "blocked_sessions": ":blocked_sessions",
                "queue_depth": ":queue_depth",
                "service_time_ms": ":service_time_ms",
                "reservation_endpoint": ":reservation_endpoint",
                "heartbeat_expires_at": ":heartbeat_expires",
                "allocation_key": ":allocation_key",
                "allocation_sort": ":allocation_sort",
                "ttl": ":ttl",
            }
            for field, placeholder in field_values.items():
                item[field] = values[placeholder]
            self.order.append(f"slot:{item['slot_index']['N']}")

        def transact_write_items(self, *, TransactItems):
            retirement_updates = [
                entry["Update"]
                for entry in TransactItems
                if "Update" in entry
                and "REMOVE allocation_key" in entry["Update"]["UpdateExpression"]
            ]
            worker_puts = [
                entry["Put"]
                for entry in TransactItems
                if "Put" in entry
                and entry["Put"]["Item"]["pk"]["S"].startswith("WORKER#")
            ]
            is_retirement = (
                bool(retirement_updates)
                and len(worker_puts) == 1
                and len(retirement_updates) + 1 == len(TransactItems)
            )
            if is_retirement:
                worker_put = worker_puts[0]
                snapshot_values = worker_put["ExpressionAttributeValues"]
                worker_id = worker_put["Item"]["pk"]["S"].removeprefix("WORKER#")
                current_worker = self.workers[worker_id]
                if (
                    (
                        ":previous_worker_epoch" in snapshot_values
                        and current_worker.get("worker_epoch")
                        != snapshot_values[":previous_worker_epoch"]
                    )
                    or (
                        ":previous_worker_epoch" not in snapshot_values
                        and "worker_epoch" in current_worker
                    )
                    or current_worker["capacity"]
                    != snapshot_values[":previous_capacity"]
                    or current_worker["role"] != snapshot_values[":previous_role"]
                    or (
                        ":previous_heartbeat_generation" in snapshot_values
                        and current_worker.get("heartbeat_generation")
                        != snapshot_values[":previous_heartbeat_generation"]
                    )
                    or (
                        ":previous_heartbeat_generation" not in snapshot_values
                        and "heartbeat_generation" in current_worker
                    )
                ):
                    raise TransactionCanceledException("stale worker snapshot")
                for update in retirement_updates:
                    item = self.slots[update["Key"]["pk"]["S"]]
                    values = update["ExpressionAttributeValues"]
                    expected_epoch = values.get(":previous_worker_epoch")
                    if (expected_epoch is None and "worker_epoch" in item) or (
                        expected_epoch is not None
                        and item.get("worker_epoch") not in (None, expected_epoch)
                    ):
                        raise TransactionCanceledException("stale slot epoch")
                    already_retired = (
                        item.get("capacity") == values[":capacity"]
                        and item.get("lifecycle") == values[":failed"]
                        and "allocation_key" not in item
                        and "allocation_sort" not in item
                    )
                    if (
                        "capacity" in item
                        and item["capacity"] != values[":previous_capacity"]
                        and not already_retired
                    ):
                        raise TransactionCanceledException("stale slot capacity")
                self.retire_attempts += 1
                self.order.append("retire-attempt")
                if self.retire_attempts <= self.retire_failures:
                    raise TransactionCanceledException("transient conflict")
                self.workers[worker_id] = dict(worker_put["Item"])
                for update in retirement_updates:
                    item = self.slots[update["Key"]["pk"]["S"]]
                    values = update["ExpressionAttributeValues"]
                    item["lifecycle"] = values[":failed"]
                    item["heartbeat_expires_at"] = values[":now"]
                    item["heartbeat_generation"] = values[":heartbeat_generation"]
                    item["capacity"] = values[":capacity"]
                    item["ttl"] = values[":ttl"]
                    item.pop("allocation_key", None)
                    item.pop("allocation_sort", None)
                self.order.append(
                    f"atomic-put:{worker_put['Item']['capacity']['N']}+retire"
                )
                return

            checks = [
                entry["ConditionCheck"]
                for entry in TransactItems
                if "ConditionCheck" in entry
            ]
            publication_updates = [
                entry["Update"]
                for entry in TransactItems
                if "Update" in entry
                and "SET item_type" in entry["Update"]["UpdateExpression"]
            ]
            if len(checks) == 1 and publication_updates:
                check = checks[0]
                worker_id = check["Key"]["pk"]["S"].removeprefix("WORKER#")
                expected_generation = check["ExpressionAttributeValues"][
                    ":heartbeat_generation"
                ]
                if (
                    self.workers[worker_id].get("heartbeat_generation")
                    != expected_generation
                ):
                    raise TransactionCanceledException("stale heartbeat publication")
                for publication_update in publication_updates:
                    self.update_item(**publication_update)
                return
            if checks:
                self.acquire_transaction = TransactItems
                for check in checks:
                    worker_id = check["Key"]["pk"]["S"].removeprefix("WORKER#")
                    worker = self.workers[worker_id]
                    values = check["ExpressionAttributeValues"]
                    if (
                        int(worker["capacity"]["N"]) <= int(values[":slot_index"]["N"])
                        or worker["worker_epoch"] != values[":worker_epoch"]
                        or worker["lifecycle"] != values[":ready"]
                        or int(worker["heartbeat_expires_at"]["N"])
                        <= int(values[":now"]["N"])
                    ):
                        raise TransactionCanceledException("worker fence rejected slot")
                return

            # Release keeps the retired slot record but removes lease ownership.
            for entry in TransactItems:
                if "Update" not in entry:
                    continue
                item = self.slots[entry["Update"]["Key"]["pk"]["S"]]
                for field in (
                    "lease_token",
                    "user_id",
                    "session_id",
                    "generation_id",
                    "lease_expires_at",
                ):
                    item.pop(field, None)

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        wall_clock=lambda: 100,
        client=client,
    )

    store._heartbeat_sync(_heartbeat("denoiser-a", "denoiser", capacity=1))

    retired = client.slots["SLOT#denoiser#denoiser-a#0001"]
    assert client.retire_attempts == 2
    assert client.order[:3] == [
        "retire-attempt",
        "retire-attempt",
        "atomic-put:1+retire",
    ]
    assert client.workers["denoiser-a"]["capacity"] == {"N": "1"}
    assert retired["lifecycle"] == {"S": "failed"}
    assert retired["heartbeat_expires_at"] == {"N": "100"}
    assert retired["capacity"] == {"N": "1"}
    assert "allocation_key" not in retired
    assert retired["lease_token"] == {"S": "active-token"}

    stale_slot = store._slot_from_item(retired)
    vae_slot = store._slot_from_item(client.slots["SLOT#vae#vae-a#0000"])
    assert (
        store._try_acquire_pair_sync(
            client,
            user_id="user-b",
            session_id="session-b",
            generation_id="generation-b",
            denoiser=stale_slot,
            vae=vae_slot,
            now_epoch=100,
            expires_epoch=160,
        )
        is None
    )
    denoiser_check = next(
        entry["ConditionCheck"]
        for entry in client.acquire_transaction
        if entry.get("ConditionCheck", {}).get("Key", {}).get("pk", {}).get("S")
        == "WORKER#denoiser-a"
    )
    assert "#capacity > :slot_index" in denoiser_check["ConditionExpression"]
    assert (
        "heartbeat_generation = :heartbeat_generation"
        in denoiser_check["ConditionExpression"]
    )

    assignment = SessionAssignment(
        user_id="user-a",
        session_id="session-a",
        generation_id="generation-a",
        token="active-token",
        expires_at=time.monotonic() + 30,
        denoiser=stale_slot,
        vae=vae_slot,
    )
    store._release_sync(assignment)
    assert "lease_token" not in retired
    assert retired["lifecycle"] == {"S": "failed"}

    store._heartbeat_sync(_heartbeat("denoiser-a", "denoiser", capacity=2))
    restored = client.slots["SLOT#denoiser#denoiser-a#0001"]
    assert restored["lifecycle"] == {"S": "ready"}
    assert restored["capacity"] == {"N": "2"}
    assert restored["allocation_key"] == {"S": "DENOISER#minwm-r1"}

    permanently_conflicted = FakeClient(retire_failures=3)
    failed_store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        wall_clock=lambda: 100,
        client=permanently_conflicted,
    )
    with pytest.raises(TransactionCanceledException, match="transient conflict"):
        failed_store._heartbeat_sync(_heartbeat("denoiser-a", "denoiser", capacity=1))
    assert permanently_conflicted.put_calls == 0
    assert permanently_conflicted.workers["denoiser-a"]["capacity"] == {"N": "2"}
    assert all(
        slot["lifecycle"] == {"S": "ready"}
        for pk, slot in permanently_conflicted.slots.items()
        if pk.startswith("SLOT#denoiser#")
    )

    # Recover the exact state left by the previous two-phase implementation if
    # it crashed after retiring slots but before publishing the worker record.
    crash_recovery = FakeClient(retire_failures=0)
    partially_retired = crash_recovery.slots["SLOT#denoiser#denoiser-a#0001"]
    partially_retired["lifecycle"] = {"S": "failed"}
    partially_retired["heartbeat_expires_at"] = {"N": "100"}
    partially_retired["capacity"] = {"N": "1"}
    partially_retired.pop("allocation_key")
    partially_retired.pop("allocation_sort")
    recovery_store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        wall_clock=lambda: 100,
        client=crash_recovery,
    )
    recovery_store._heartbeat_sync(_heartbeat("denoiser-a", "denoiser", capacity=1))
    assert crash_recovery.workers["denoiser-a"]["capacity"] == {"N": "1"}
    assert partially_retired["lifecycle"] == {"S": "failed"}

    # Legacy records written before worker_epoch was introduced migrate on the
    # first shrinking heartbeat instead of becoming permanently unshrinkable.
    legacy = FakeClient(retire_failures=0)
    legacy.workers["denoiser-a"].pop("worker_epoch")
    for pk, slot in legacy.slots.items():
        if pk.startswith("SLOT#denoiser#"):
            slot.pop("worker_epoch")
    legacy_store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        wall_clock=lambda: 100,
        client=legacy,
    )
    legacy_store._heartbeat_sync(
        _heartbeat(
            "denoiser-a",
            "denoiser",
            capacity=1,
            worker_epoch="epoch-migrated",
        )
    )
    assert legacy.workers["denoiser-a"]["worker_epoch"] == {"S": "epoch-migrated"}
    assert legacy.slots["SLOT#denoiser#denoiser-a#0001"]["lifecycle"] == {"S": "failed"}

    class RacingClient(FakeClient):
        def __init__(self):
            super().__init__(retire_failures=0)
            self.store = None
            self.injected_new_epoch = False

        def transact_write_items(self, *, TransactItems):
            shrinking_worker = next(
                (
                    entry["Put"]["Item"]
                    for entry in TransactItems
                    if "Put" in entry
                    and entry["Put"]["Item"].get("capacity") == {"N": "1"}
                ),
                None,
            )
            if shrinking_worker is not None and not self.injected_new_epoch:
                self.injected_new_epoch = True
                self.store._heartbeat_sync(
                    _heartbeat(
                        "denoiser-a",
                        "denoiser",
                        capacity=3,
                        worker_epoch="epoch-b",
                    )
                )
            return super().transact_write_items(TransactItems=TransactItems)

    racing = RacingClient()
    racing_store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        wall_clock=lambda: 100,
        client=racing,
    )
    racing.store = racing_store

    with pytest.raises(TransactionCanceledException, match="stale worker"):
        racing_store._heartbeat_sync(
            _heartbeat(
                "denoiser-a",
                "denoiser",
                capacity=1,
                worker_epoch="epoch-a",
            )
        )

    # The stale epoch-a shrink cannot overwrite either the epoch-b worker
    # heartbeat or the slots restored/created by its grow.
    assert racing.workers["denoiser-a"]["worker_epoch"] == {"S": "epoch-b"}
    assert racing.workers["denoiser-a"]["capacity"] == {"N": "3"}
    assert {
        (
            racing.slots[f"SLOT#denoiser#denoiser-a#{index:04d}"]["worker_epoch"]["S"],
            racing.slots[f"SLOT#denoiser#denoiser-a#{index:04d}"]["capacity"]["N"],
            racing.slots[f"SLOT#denoiser#denoiser-a#{index:04d}"]["lifecycle"]["S"],
        )
        for index in range(3)
    } == {("epoch-b", "3", "ready")}


def test_dynamodb_atomic_shrink_recovers_partial_retire_and_fences_racing_grow():
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")

    with moto.mock_aws():
        table_name = "minwm-realtime-coordinator"
        client = boto3.client("dynamodb", region_name="us-east-1")
        _create_dynamodb_coordinator_table(client, table_name)

        store = DynamoDBCoordinatorStore(
            table_name,
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 100,
            client=client,
        )
        store._heartbeat_sync(_heartbeat("denoiser-partial", "denoiser", capacity=2))

        worker_key = {
            "pk": {"S": "WORKER#denoiser-partial"},
            "sk": {"S": "HEARTBEAT"},
        }
        client.update_item(
            TableName=table_name,
            Key=worker_key,
            UpdateExpression="REMOVE heartbeat_generation",
        )
        for slot_index in range(2):
            client.update_item(
                TableName=table_name,
                Key={
                    "pk": {"S": f"SLOT#denoiser#denoiser-partial#{slot_index:04d}"},
                    "sk": {"S": "LEASE"},
                },
                UpdateExpression="REMOVE heartbeat_generation",
            )

        partial_slot_key = {
            "pk": {"S": "SLOT#denoiser#denoiser-partial#0001"},
            "sk": {"S": "LEASE"},
        }
        client.update_item(
            TableName=table_name,
            Key=partial_slot_key,
            UpdateExpression=(
                "SET #lifecycle = :failed, heartbeat_expires_at = :now, "
                "#capacity = :capacity, #ttl = :ttl "
                "REMOVE allocation_key, allocation_sort"
            ),
            ExpressionAttributeNames={
                "#capacity": "capacity",
                "#lifecycle": "lifecycle",
                "#ttl": "ttl",
            },
            ExpressionAttributeValues={
                ":failed": {"S": "failed"},
                ":now": {"N": "100"},
                ":capacity": {"N": "1"},
                ":ttl": {"N": "86500"},
            },
        )

        # This is the old two-phase crash state: the slot retirement committed,
        # but the worker heartbeat still advertises the previous capacity.
        assert client.get_item(
            TableName=table_name,
            Key=worker_key,
            ConsistentRead=True,
        )["Item"]["capacity"] == {"N": "2"}

        store._heartbeat_sync(_heartbeat("denoiser-partial", "denoiser", capacity=1))
        recovered_worker = client.get_item(
            TableName=table_name,
            Key=worker_key,
            ConsistentRead=True,
        )["Item"]
        recovered_slot = client.get_item(
            TableName=table_name,
            Key=partial_slot_key,
            ConsistentRead=True,
        )["Item"]
        assert recovered_worker["capacity"] == {"N": "1"}
        assert recovered_slot["capacity"] == {"N": "1"}
        assert recovered_slot["lifecycle"] == {"S": "failed"}
        assert (
            recovered_slot["heartbeat_generation"]
            == recovered_worker["heartbeat_generation"]
        )
        assert "allocation_key" not in recovered_slot
        assert "allocation_sort" not in recovered_slot

        base_racing_store = DynamoDBCoordinatorStore(
            table_name,
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 100,
            client=client,
        )
        base_racing_store._heartbeat_sync(
            _heartbeat("denoiser-race", "denoiser", capacity=2)
        )

        class GrowBeforeShrinkTransaction:
            def __init__(self):
                self.injected = False

            def __getattr__(self, name):
                return getattr(client, name)

            def transact_write_items(self, **kwargs):
                if not self.injected:
                    self.injected = True
                    base_racing_store._heartbeat_sync(
                        _heartbeat(
                            "denoiser-race",
                            "denoiser",
                            capacity=3,
                            worker_epoch="epoch-b",
                        )
                    )
                return client.transact_write_items(**kwargs)

        racing_client = GrowBeforeShrinkTransaction()
        stale_shrink_store = DynamoDBCoordinatorStore(
            table_name,
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 100,
            client=racing_client,
        )
        with pytest.raises(client.exceptions.TransactionCanceledException):
            stale_shrink_store._heartbeat_sync(
                _heartbeat("denoiser-race", "denoiser", capacity=1)
            )

        raced_worker = client.get_item(
            TableName=table_name,
            Key={
                "pk": {"S": "WORKER#denoiser-race"},
                "sk": {"S": "HEARTBEAT"},
            },
            ConsistentRead=True,
        )["Item"]
        assert raced_worker["worker_epoch"] == {"S": "epoch-b"}
        assert raced_worker["capacity"] == {"N": "3"}
        for slot_index in range(3):
            raced_slot = client.get_item(
                TableName=table_name,
                Key={
                    "pk": {"S": f"SLOT#denoiser#denoiser-race#{slot_index:04d}"},
                    "sk": {"S": "LEASE"},
                },
                ConsistentRead=True,
            )["Item"]
            assert raced_slot["worker_epoch"] == {"S": "epoch-b"}
            assert raced_slot["capacity"] == {"N": "3"}
            assert raced_slot["lifecycle"] == {"S": "ready"}
            assert raced_slot["allocation_key"] == {"S": "DENOISER#minwm-r1"}


def test_dynamodb_generation_fences_delayed_same_epoch_slot_publication():
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")

    with moto.mock_aws():
        table_name = "minwm-delayed-heartbeat-coordinator"
        client = boto3.client("dynamodb", region_name="us-east-1")
        _create_dynamodb_coordinator_table(client, table_name)
        base_store = DynamoDBCoordinatorStore(
            table_name,
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 100,
            client=client,
        )
        heartbeat = _heartbeat("denoiser-delayed", "denoiser", capacity=2)
        base_store._heartbeat_sync(heartbeat)

        class DelaySecondSlotPublication:
            def __init__(self):
                self.delayed_transaction = None

            def __getattr__(self, name):
                return getattr(client, name)

            def transact_write_items(self, **kwargs):
                slot_updates = [
                    item["Update"]
                    for item in kwargs["TransactItems"]
                    if "Update" in item
                    and "SET item_type" in item["Update"]["UpdateExpression"]
                ]
                if self.delayed_transaction is None and any(
                    update["Key"]["pk"]["S"].endswith("#0001")
                    for update in slot_updates
                ):
                    self.delayed_transaction = kwargs
                    return {}
                return client.transact_write_items(**kwargs)

        delaying_client = DelaySecondSlotPublication()
        delayed_store = DynamoDBCoordinatorStore(
            table_name,
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 101,
            client=delaying_client,
        )
        delayed_store._heartbeat_sync(heartbeat)
        assert delaying_client.delayed_transaction is not None
        delayed_generation = next(
            item["ConditionCheck"]["ExpressionAttributeValues"][
                ":heartbeat_generation"
            ]["S"]
            for item in delaying_client.delayed_transaction["TransactItems"]
            if "ConditionCheck" in item
        )

        shrink_store = DynamoDBCoordinatorStore(
            table_name,
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 102,
            client=client,
        )
        shrink_store._heartbeat_sync(
            _heartbeat("denoiser-delayed", "denoiser", capacity=1)
        )
        current_worker = client.get_item(
            TableName=table_name,
            Key={
                "pk": {"S": "WORKER#denoiser-delayed"},
                "sk": {"S": "HEARTBEAT"},
            },
            ConsistentRead=True,
        )["Item"]
        assert current_worker["capacity"] == {"N": "1"}
        assert current_worker["heartbeat_generation"]["S"] != delayed_generation

        with pytest.raises(client.exceptions.TransactionCanceledException):
            client.transact_write_items(**delaying_client.delayed_transaction)

        retired_slot = client.get_item(
            TableName=table_name,
            Key={
                "pk": {"S": "SLOT#denoiser#denoiser-delayed#0001"},
                "sk": {"S": "LEASE"},
            },
            ConsistentRead=True,
        )["Item"]
        assert retired_slot["capacity"] == {"N": "1"}
        assert retired_slot["lifecycle"] == {"S": "failed"}
        assert (
            retired_slot["heartbeat_generation"]
            == current_worker["heartbeat_generation"]
        )
        assert "allocation_key" not in retired_slot
        assert "allocation_sort" not in retired_slot

        allocatable = client.query(
            TableName=table_name,
            IndexName="allocation-index",
            KeyConditionExpression="allocation_key = :allocation",
            ExpressionAttributeValues={":allocation": {"S": "DENOISER#minwm-r1"}},
        )["Items"]
        assert {
            item["slot_index"]["N"]
            for item in allocatable
            if item.get("worker_id") == {"S": "denoiser-delayed"}
        } == {"0"}

        shrink_store._heartbeat_sync(_heartbeat("vae-delayed", "vae", capacity=1))
        assignment = shrink_store._acquire_sync(
            user_id="user-delayed",
            session_id="session-delayed",
            generation_id="generation-delayed",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert assignment.denoiser.worker_id == "denoiser-delayed"
        assert assignment.denoiser.slot_index == 0


def test_dynamodb_epoch_change_partial_slot_publication_self_heals():
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")

    with moto.mock_aws():
        table_name = "minwm-epoch-heal-coordinator"
        client = boto3.client("dynamodb", region_name="us-east-1")
        _create_dynamodb_coordinator_table(client, table_name)
        worker_id = "denoiser-epoch-heal"
        slot_key = {
            "pk": {"S": f"SLOT#denoiser#{worker_id}#0000"},
            "sk": {"S": "LEASE"},
        }
        worker_key = {
            "pk": {"S": f"WORKER#{worker_id}"},
            "sk": {"S": "HEARTBEAT"},
        }
        epoch_a = _heartbeat(
            worker_id,
            "denoiser",
            capacity=1,
            worker_epoch="epoch-a",
        )
        base_store = DynamoDBCoordinatorStore(
            table_name,
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 100,
            client=client,
        )
        base_store._heartbeat_sync(epoch_a)
        client.update_item(
            TableName=table_name,
            Key=slot_key,
            UpdateExpression=(
                "SET lease_token = :token, user_id = :user, "
                "session_id = :session, generation_id = :generation, "
                "lease_expires_at = :expires"
            ),
            ExpressionAttributeValues={
                ":token": {"S": "active-token"},
                ":user": {"S": "active-user"},
                ":session": {"S": "active-session"},
                ":generation": {"S": "active-generation"},
                ":expires": {"N": "180"},
            },
        )

        class RejectActiveSlotPublication:
            def __init__(self):
                self.rejected_attempts = 0

            def __getattr__(self, name):
                return getattr(client, name)

            def transact_write_items(self, **kwargs):
                is_slot_publication = any(
                    "Update" in item
                    and "SET item_type" in item["Update"]["UpdateExpression"]
                    for item in kwargs["TransactItems"]
                )
                if is_slot_publication:
                    self.rejected_attempts += 1
                    raise client.exceptions.TransactionCanceledException(
                        {
                            "Error": {
                                "Code": "TransactionCanceledException",
                                "Message": "injected active slot publication failure",
                            }
                        },
                        "TransactWriteItems",
                    )
                return client.transact_write_items(**kwargs)

        rejecting_client = RejectActiveSlotPublication()
        failed_epoch_change = DynamoDBCoordinatorStore(
            table_name,
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 101,
            client=rejecting_client,
        )
        epoch_b = _heartbeat(
            worker_id,
            "denoiser",
            capacity=1,
            worker_epoch="epoch-b",
        )
        with pytest.raises(client.exceptions.TransactionCanceledException):
            failed_epoch_change._heartbeat_sync(epoch_b)
        assert rejecting_client.rejected_attempts == 3

        partial_worker = client.get_item(
            TableName=table_name,
            Key=worker_key,
            ConsistentRead=True,
        )["Item"]
        partial_slot = client.get_item(
            TableName=table_name,
            Key=slot_key,
            ConsistentRead=True,
        )["Item"]
        assert partial_worker["worker_epoch"] == {"S": "epoch-b"}
        assert partial_slot["worker_epoch"] == {"S": "epoch-a"}
        assert (
            partial_worker["heartbeat_generation"]
            != partial_slot["heartbeat_generation"]
        )
        assert partial_slot["lease_token"] == {"S": "active-token"}

        healing_store = DynamoDBCoordinatorStore(
            table_name,
            ttl_s=60,
            worker_ttl_s=30,
            wall_clock=lambda: 102,
            client=client,
        )
        healing_store._heartbeat_sync(epoch_b)
        healed_worker = client.get_item(
            TableName=table_name,
            Key=worker_key,
            ConsistentRead=True,
        )["Item"]
        healed_slot = client.get_item(
            TableName=table_name,
            Key=slot_key,
            ConsistentRead=True,
        )["Item"]
        assert healed_slot["worker_epoch"] == {"S": "epoch-b"}
        assert (
            healed_slot["heartbeat_generation"] == healed_worker["heartbeat_generation"]
        )
        assert healed_slot["lease_token"] == {"S": "active-token"}
        assert healed_slot["lifecycle"] == {"S": "ready"}
        assert healed_slot["allocation_key"] == {"S": "DENOISER#minwm-r1"}


def test_dynamodb_renew_condition_checks_current_worker_epochs_and_expiry():
    class TransactionCanceledException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.transaction = None

        def transact_write_items(self, *, TransactItems):
            self.transaction = TransactItems

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
    )
    assignment = SessionAssignment(
        user_id="user-a",
        session_id="session-a",
        generation_id="generation-a",
        token="token-a",
        expires_at=time.monotonic() + 30,
        denoiser=WorkerSlot(
            worker_id="denoiser-a",
            role="denoiser",
            endpoint="ws://denoiser-a/generate",
            az="us-east-2a",
            slot_index=0,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            worker_epoch="denoiser-epoch",
        ),
        vae=WorkerSlot(
            worker_id="vae-a",
            role="vae",
            endpoint="ws://vae-a/decode",
            az="us-east-2a",
            slot_index=0,
            model_revision="all",
            vae_fingerprint="taew2_2",
            worker_epoch="vae-epoch",
        ),
    )

    store._renew_sync(assignment)

    checks = [
        item["ConditionCheck"]
        for item in client.transaction
        if "ConditionCheck" in item
    ]
    assert len(checks) == 2
    assert {check["Key"]["pk"]["S"] for check in checks} == {
        "WORKER#denoiser-a",
        "WORKER#vae-a",
    }
    epochs = {
        check["ExpressionAttributeValues"][":worker_epoch"]["S"] for check in checks
    }
    assert epochs == {"denoiser-epoch", "vae-epoch"}
    assert all(
        "heartbeat_expires_at > :now" in check["ConditionExpression"]
        for check in checks
    )


def test_dynamodb_renew_retries_a_transient_transaction_conflict():
    class TransactionCanceledException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException

    class FakeClient:
        exceptions = FakeExceptions()

        def __init__(self):
            self.transactions = 0

        def transact_write_items(self, **_kwargs):
            self.transactions += 1
            if self.transactions < 5:
                raise TransactionCanceledException("heartbeat write conflict")

        def get_item(self, *, Key, **_kwargs):
            pk = Key["pk"]["S"]
            if pk.startswith("WORKER#"):
                worker_id = pk.removeprefix("WORKER#")
                return {
                    "Item": {
                        "worker_epoch": {"S": f"{worker_id}-epoch"},
                        "heartbeat_expires_at": {"N": "9999999999"},
                        "lifecycle": {"S": "ready"},
                    }
                }
            return {"Item": {"lease_token": {"S": "token-a"}}}

    client = FakeClient()
    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=client,
    )
    assignment = SessionAssignment(
        user_id="user-a",
        session_id="session-a",
        generation_id="generation-a",
        token="token-a",
        expires_at=time.monotonic() + 30,
        denoiser=WorkerSlot(
            "denoiser-a",
            "denoiser",
            "ws://denoiser-a/generate",
            "us-east-2a",
            0,
            "minwm-r1",
            "taew2_2",
            worker_epoch="denoiser-a-epoch",
        ),
        vae=WorkerSlot(
            "vae-a",
            "vae",
            "ws://vae-a/decode",
            "us-east-2a",
            0,
            "all",
            "taew2_2",
            worker_epoch="vae-a-epoch",
        ),
    )

    renewed = store._renew_sync(assignment)

    assert renewed.expires_at > assignment.expires_at
    assert client.transactions == 5


def test_dynamodb_renew_classifies_failed_worker_as_worker_lost():
    class TransactionCanceledException(Exception):
        pass

    class FakeExceptions:
        pass

    FakeExceptions.TransactionCanceledException = TransactionCanceledException

    class FakeClient:
        exceptions = FakeExceptions()

        def transact_write_items(self, **_kwargs):
            raise TransactionCanceledException("worker failed")

        def get_item(self, *, Key, **_kwargs):
            worker_id = Key["pk"]["S"].removeprefix("WORKER#")
            return {
                "Item": {
                    "worker_epoch": {"S": f"{worker_id}-epoch"},
                    "heartbeat_expires_at": {"N": "9999999999"},
                    "lifecycle": {"S": "failed"},
                }
            }

    store = DynamoDBCoordinatorStore(
        "minwm-realtime-coordinator",
        ttl_s=60,
        worker_ttl_s=30,
        client=FakeClient(),
    )
    assignment = SessionAssignment(
        user_id="user-a",
        session_id="session-a",
        generation_id="generation-a",
        token="token-a",
        expires_at=time.monotonic() + 30,
        denoiser=WorkerSlot(
            "denoiser-a",
            "denoiser",
            "ws://denoiser-a/generate",
            "us-east-2a",
            0,
            "minwm-r1",
            "taew2_2",
            worker_epoch="denoiser-a-epoch",
        ),
        vae=WorkerSlot(
            "vae-a",
            "vae",
            "ws://vae-a/decode",
            "us-east-2a",
            0,
            "all",
            "taew2_2",
            worker_epoch="vae-a-epoch",
        ),
    )

    with pytest.raises(CoordinatorRejected, match="WORKER_LOST"):
        store._renew_sync(assignment)


def test_coordinator_cancellation_compensates_assignment_and_partial_reservations():
    class ReservationClient:
        def __init__(self):
            self.vae_started = asyncio.Event()
            self.never = asyncio.Event()
            self.released = []

        async def reserve(self, slot, **_identity):
            if slot.role == "vae":
                self.vae_started.set()
                await self.never.wait()

        async def release(self, slot, *, token):
            self.released.append((slot.role, token))

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        reservations = ReservationClient()
        coordinator = RealtimeCoordinator(
            store,
            wait_timeout_s=5,
            reservation_client=reservations,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))

        task = asyncio.create_task(
            coordinator.admit(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )
        )
        await reservations.vae_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert {role for role, _token in reservations.released} == {
            "denoiser",
            "vae",
        }
        replacement = await store.acquire(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"
        assert (await coordinator.capacity_snapshot())["roles"]["denoiser"][
            "waiting_sessions"
        ] == 0

    asyncio.run(run())


def test_coordinator_compensation_retries_transient_worker_release_failures():
    class ReservationClient:
        def __init__(self):
            self.release_attempts = {"denoiser": 0, "vae": 0}

        async def reserve(self, slot, **_identity):
            if slot.role == "vae":
                raise RuntimeError("reserve failed")

        async def release(self, slot, *, token):
            del token
            self.release_attempts[slot.role] += 1
            if slot.role == "denoiser" and self.release_attempts[slot.role] < 3:
                raise RuntimeError("transient release failure")

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        reservations = ReservationClient()
        coordinator = RealtimeCoordinator(
            store,
            wait_timeout_s=0.25,
            reservation_client=reservations,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))

        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coordinator.admit(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
                wait_for_capacity=False,
            )

        assert reservations.release_attempts["denoiser"] == 3
        assert reservations.release_attempts["vae"] == 1
        replacement = await store.acquire(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"

    asyncio.run(run())


def test_coordinator_deadline_covers_worker_reserve_and_compensates_assignment():
    class ReservationClient:
        def __init__(self):
            self.released = []

        async def reserve(self, slot, **_identity):
            if slot.role == "vae":
                await asyncio.sleep(1)

        async def release(self, slot, *, token):
            self.released.append((slot.role, token))

    async def run():
        store = InMemoryCoordinatorStore(ttl_s=60, worker_ttl_s=30)
        reservations = ReservationClient()
        coordinator = RealtimeCoordinator(
            store,
            wait_timeout_s=0.05,
            reservation_client=reservations,
        )
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))

        started = time.monotonic()
        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coordinator.admit(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )
        elapsed = time.monotonic() - started

        assert elapsed < 0.2
        assert {role for role, _token in reservations.released} == {
            "denoiser",
            "vae",
        }
        replacement = await store.acquire(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"

    asyncio.run(run())


def test_coordinator_deadline_returns_before_slow_acquire_and_cleans_late_commit():
    class SlowCommitStore(InMemoryCoordinatorStore):
        async def acquire(self, **request):
            await asyncio.sleep(0.05)
            return await super().acquire(**request)

    async def run():
        store = SlowCommitStore(ttl_s=60, worker_ttl_s=30)
        coordinator = RealtimeCoordinator(store, wait_timeout_s=0.01)
        await coordinator.heartbeat(_heartbeat("denoiser-a", "denoiser"))
        await coordinator.heartbeat(_heartbeat("vae-a", "vae"))

        started = time.monotonic()
        with pytest.raises(CoordinatorRejected, match="CAPACITY_EXHAUSTED"):
            await coordinator.admit(
                user_id="user-a",
                session_id="session-a",
                generation_id="generation-a",
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            )
        assert time.monotonic() - started < 0.04

        await asyncio.sleep(0.08)
        replacement = await store.acquire(
            user_id="user-a",
            session_id="session-b",
            generation_id="generation-b",
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
        )
        assert replacement.session_id == "session-b"

    asyncio.run(run())
