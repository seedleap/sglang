apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: zing-denoiser-5090-sp2
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-denoiser-5090-sp2
    app.kubernetes.io/part-of: minwm-realtime
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-denoiser-5090-sp2
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
  template:
    metadata:
      annotations:
        seedleap.ai/runtime-source-patch: ${RUNTIME_SOURCE_PATCH_VERSION}
      labels:
        app.kubernetes.io/name: zing-denoiser-5090-sp2
        app.kubernetes.io/part-of: minwm-realtime
        seedleap.ai/worker-role: denoiser
        seedleap.ai/gpu-topology: 7p1-sp1
    spec:
      serviceAccountName: zing-realtime
      runtimeClassName: nvidia
      terminationGracePeriodSeconds: 120
      nodeSelector:
        seedleap.ai/gpu-pool: aliyun-beijing-5090
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      initContainers:
        - name: denoiser-heartbeat-0
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:30000/health
              --state-url=http://127.0.0.1:30000/v1/realtime_worker/state
              --worker-id=$(NODE_NAME)-denoiser-0
              --worker-epoch-file=/var/run/minwm-worker/epoch
              --role=denoiser
              --endpoint=ws://$(POD_IP):30000/v1/realtime_video/generate
              --reservation-endpoint=http://$(POD_IP):30000/v1/realtime_worker
              --node-name=$(NODE_NAME)
              --capacity=1
              --model-revision=wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2
              --vae-fingerprint=taew2_2-d053e216
              --interval-s=5
          env:
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: void}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          resources:
            requests: {cpu: 25m, memory: 64Mi}
            limits: {cpu: 200m, memory: 256Mi}
          volumeMounts:
            - {name: worker-epoch-denoiser-0, mountPath: /var/run/minwm-worker}
        - name: denoiser-heartbeat-1
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:30010/health
              --state-url=http://127.0.0.1:30010/v1/realtime_worker/state
              --worker-id=$(NODE_NAME)-denoiser-1
              --worker-epoch-file=/var/run/minwm-worker/epoch
              --role=denoiser
              --endpoint=ws://$(POD_IP):30010/v1/realtime_video/generate
              --reservation-endpoint=http://$(POD_IP):30010/v1/realtime_worker
              --node-name=$(NODE_NAME)
              --capacity=1
              --model-revision=wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2
              --vae-fingerprint=taew2_2-d053e216
              --interval-s=5
          env:
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: void}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          resources:
            requests: {cpu: 25m, memory: 64Mi}
            limits: {cpu: 200m, memory: 256Mi}
          volumeMounts:
            - {name: worker-epoch-denoiser-1, mountPath: /var/run/minwm-worker}
        - name: denoiser-heartbeat-2
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:30020/health
              --state-url=http://127.0.0.1:30020/v1/realtime_worker/state
              --worker-id=$(NODE_NAME)-denoiser-2
              --worker-epoch-file=/var/run/minwm-worker/epoch
              --role=denoiser
              --endpoint=ws://$(POD_IP):30020/v1/realtime_video/generate
              --reservation-endpoint=http://$(POD_IP):30020/v1/realtime_worker
              --node-name=$(NODE_NAME)
              --capacity=1
              --model-revision=wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2
              --vae-fingerprint=taew2_2-d053e216
              --interval-s=5
          env:
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: void}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          resources:
            requests: {cpu: 25m, memory: 64Mi}
            limits: {cpu: 200m, memory: 256Mi}
          volumeMounts:
            - {name: worker-epoch-denoiser-2, mountPath: /var/run/minwm-worker}
        - name: denoiser-heartbeat-3
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:30030/health
              --state-url=http://127.0.0.1:30030/v1/realtime_worker/state
              --worker-id=$(NODE_NAME)-denoiser-3
              --worker-epoch-file=/var/run/minwm-worker/epoch
              --role=denoiser
              --endpoint=ws://$(POD_IP):30030/v1/realtime_video/generate
              --reservation-endpoint=http://$(POD_IP):30030/v1/realtime_worker
              --node-name=$(NODE_NAME)
              --capacity=1
              --model-revision=wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2
              --vae-fingerprint=taew2_2-d053e216
              --interval-s=5
          env:
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: void}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          resources:
            requests: {cpu: 25m, memory: 64Mi}
            limits: {cpu: 200m, memory: 256Mi}
          volumeMounts:
            - {name: worker-epoch-denoiser-3, mountPath: /var/run/minwm-worker}
        - name: denoiser-heartbeat-4
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:30040/health
              --state-url=http://127.0.0.1:30040/v1/realtime_worker/state
              --worker-id=$(NODE_NAME)-denoiser-4
              --worker-epoch-file=/var/run/minwm-worker/epoch
              --role=denoiser
              --endpoint=ws://$(POD_IP):30040/v1/realtime_video/generate
              --reservation-endpoint=http://$(POD_IP):30040/v1/realtime_worker
              --node-name=$(NODE_NAME)
              --capacity=1
              --model-revision=wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2
              --vae-fingerprint=taew2_2-d053e216
              --interval-s=5
          env:
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: void}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          resources:
            requests: {cpu: 25m, memory: 64Mi}
            limits: {cpu: 200m, memory: 256Mi}
          volumeMounts:
            - {name: worker-epoch-denoiser-4, mountPath: /var/run/minwm-worker}
        - name: denoiser-heartbeat-5
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:30050/health
              --state-url=http://127.0.0.1:30050/v1/realtime_worker/state
              --worker-id=$(NODE_NAME)-denoiser-5
              --worker-epoch-file=/var/run/minwm-worker/epoch
              --role=denoiser
              --endpoint=ws://$(POD_IP):30050/v1/realtime_video/generate
              --reservation-endpoint=http://$(POD_IP):30050/v1/realtime_worker
              --node-name=$(NODE_NAME)
              --capacity=1
              --model-revision=wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2
              --vae-fingerprint=taew2_2-d053e216
              --interval-s=5
          env:
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: void}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          resources:
            requests: {cpu: 25m, memory: 64Mi}
            limits: {cpu: 200m, memory: 256Mi}
          volumeMounts:
            - {name: worker-epoch-denoiser-5, mountPath: /var/run/minwm-worker}
        - name: denoiser-heartbeat-6
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:30060/health
              --state-url=http://127.0.0.1:30060/v1/realtime_worker/state
              --worker-id=$(NODE_NAME)-denoiser-6
              --worker-epoch-file=/var/run/minwm-worker/epoch
              --role=denoiser
              --endpoint=ws://$(POD_IP):30060/v1/realtime_video/generate
              --reservation-endpoint=http://$(POD_IP):30060/v1/realtime_worker
              --node-name=$(NODE_NAME)
              --capacity=1
              --model-revision=wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2
              --vae-fingerprint=taew2_2-d053e216
              --interval-s=5
          env:
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: void}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          resources:
            requests: {cpu: 25m, memory: 64Mi}
            limits: {cpu: 200m, memory: 256Mi}
          volumeMounts:
            - {name: worker-epoch-denoiser-6, mountPath: /var/run/minwm-worker}
      containers:
        - name: denoiser-0
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          workingDir: /opt/sglang
          command: [/bin/bash, -lc]
          args:
            - |
              set -euo pipefail
              exec 9>/var/run/minwm-startup-lock/denoiser.lock
              flock -x 9
              python3 -m sglang.multimodal_gen.tools.minwm_profile_launcher \
                --profile auto \
                --taehv-checkpoint-path /models/taehv/taew2_2.pth \
                -- \
                --model-path /models/minwm-tianpeng-gap12 \
                --num-gpus 1 \
                --tp-size 1 \
                --sp-degree 1 \
                --ulysses-degree 1 \
                --ring-degree 1 \
                --enable-cuda-graph true \
                --batching-max-size 1 \
                --batching-delay-ms 2 \
                --realtime-max-sessions 1 \
                --realtime-max-sessions-per-worker 1 \
                --realtime-vae-backend taehv_remote \
                --realtime-vae-transport websocket \
                --realtime-session-idle-timeout-s 90 \
                --realtime-session-max-lifetime-s 70 \
                --realtime-admission-wait-s 10 \
                --host 0.0.0.0 \
                --port 30000 \
                --master-port 30200 \
                --scheduler-port 5615 \
                --strict-ports &
              child=$!
              terminate() {
                kill -TERM "${child}" 2>/dev/null || true
                wait "${child}" || true
                exit 143
              }
              trap terminate TERM INT
              until curl --fail --silent --max-time 2 http://127.0.0.1:30000/health >/dev/null; do
                if ! kill -0 "${child}" 2>/dev/null; then
                  wait "${child}"
                  exit $?
                fi
                sleep 2
              done
              flock -u 9
              wait "${child}"
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: NVIDIA_VISIBLE_DEVICES, value: "0"}
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
            - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
            - {name: SGLANG_DISABLE_PDEATHSIG, value: "1"}
            - {name: OMP_NUM_THREADS, value: "4"}
            - {name: MKL_NUM_THREADS, value: "4"}
            - {name: OPENBLAS_NUM_THREADS, value: "4"}
            - {name: NUMEXPR_NUM_THREADS, value: "4"}
            - {name: VECLIB_MAXIMUM_THREADS, value: "4"}
            - {name: TOKENIZERS_PARALLELISM, value: "false"}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
            - {name: MINWM_ATTENTION_IMPL, value: packed}
            - {name: MINWM_SEGMENT_COMPILE, value: "true"}
            - {name: MINWM_CACHE_ROTATED_K, value: "true"}
            - {name: MINWM_PRECOMPUTE_CACHE_ROPE, value: "true"}
            - {name: MINWM_CACHE_PACKED_METADATA, value: "true"}
            - {name: MINWM_PACKED_ATTENTION_DETERMINISTIC, value: "false"}
            - {name: MINWM_NATIVE_COMPONENTS, value: ""}
            - {name: MINWM_RUNTIME_ALIGNMENT_LOG, value: "1"}
            - {name: SGLANG_MINWM_REQUIRE_SM120_FA4, value: "1"}
            - {name: SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D, value: "false"}
            - {name: NCCL_DEBUG, value: WARN}
            - {name: NCCL_PROTO, value: Simple}
            - {name: REALTIME_MAX_SESSIONS, value: "1"}
            - {name: REALTIME_UI_CONFIG_JSON, value: '{"targetFps":${REALTIME_TARGET_FPS},"size":"${REALTIME_SIZE}","dualModels":{"minwm":{"sinkSize":8,"windowFrames":32}}}'}
          ports:
            - {name: denoiser-0, containerPort: 30000}
          startupProbe:
            httpGet: {path: /health, port: denoiser-0}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 360
          readinessProbe:
            httpGet: {path: /health, port: denoiser-0}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          livenessProbe:
            httpGet: {path: /health, port: denoiser-0}
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 6
          resources:
            requests: {cpu: "16", memory: 120Gi}
            limits: {cpu: "24", memory: 180Gi}
          securityContext:
            capabilities:
              add: [SYS_ADMIN]
          volumeMounts:
            - {name: worker-epoch-denoiser-0, mountPath: /var/run/minwm-worker}
            - {name: startup-lock, mountPath: /var/run/minwm-startup-lock}
            - {name: model, mountPath: /models/minwm-tianpeng-gap12, readOnly: true}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /sgl-workspace/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: shm, mountPath: /dev/shm}
        - name: denoiser-1
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          workingDir: /opt/sglang
          command: [/bin/bash, -lc]
          args:
            - |
              set -euo pipefail
              exec 9>/var/run/minwm-startup-lock/denoiser.lock
              flock -x 9
              python3 -m sglang.multimodal_gen.tools.minwm_profile_launcher \
                --profile auto \
                --taehv-checkpoint-path /models/taehv/taew2_2.pth \
                -- \
                --model-path /models/minwm-tianpeng-gap12 \
                --num-gpus 1 \
                --tp-size 1 \
                --sp-degree 1 \
                --ulysses-degree 1 \
                --ring-degree 1 \
                --enable-cuda-graph true \
                --batching-max-size 1 \
                --batching-delay-ms 2 \
                --realtime-max-sessions 1 \
                --realtime-max-sessions-per-worker 1 \
                --realtime-vae-backend taehv_remote \
                --realtime-vae-transport websocket \
                --realtime-session-idle-timeout-s 90 \
                --realtime-session-max-lifetime-s 70 \
                --realtime-admission-wait-s 10 \
                --host 0.0.0.0 \
                --port 30010 \
                --master-port 30210 \
                --scheduler-port 5625 \
                --strict-ports &
              child=$!
              terminate() {
                kill -TERM "${child}" 2>/dev/null || true
                wait "${child}" || true
                exit 143
              }
              trap terminate TERM INT
              until curl --fail --silent --max-time 2 http://127.0.0.1:30010/health >/dev/null; do
                if ! kill -0 "${child}" 2>/dev/null; then
                  wait "${child}"
                  exit $?
                fi
                sleep 2
              done
              flock -u 9
              wait "${child}"
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: NVIDIA_VISIBLE_DEVICES, value: "1"}
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
            - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
            - {name: SGLANG_DISABLE_PDEATHSIG, value: "1"}
            - {name: OMP_NUM_THREADS, value: "4"}
            - {name: MKL_NUM_THREADS, value: "4"}
            - {name: OPENBLAS_NUM_THREADS, value: "4"}
            - {name: NUMEXPR_NUM_THREADS, value: "4"}
            - {name: VECLIB_MAXIMUM_THREADS, value: "4"}
            - {name: TOKENIZERS_PARALLELISM, value: "false"}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
            - {name: MINWM_ATTENTION_IMPL, value: packed}
            - {name: MINWM_SEGMENT_COMPILE, value: "true"}
            - {name: MINWM_CACHE_ROTATED_K, value: "true"}
            - {name: MINWM_PRECOMPUTE_CACHE_ROPE, value: "true"}
            - {name: MINWM_CACHE_PACKED_METADATA, value: "true"}
            - {name: MINWM_PACKED_ATTENTION_DETERMINISTIC, value: "false"}
            - {name: MINWM_NATIVE_COMPONENTS, value: ""}
            - {name: MINWM_RUNTIME_ALIGNMENT_LOG, value: "1"}
            - {name: SGLANG_MINWM_REQUIRE_SM120_FA4, value: "1"}
            - {name: SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D, value: "false"}
            - {name: NCCL_DEBUG, value: WARN}
            - {name: NCCL_PROTO, value: Simple}
            - {name: REALTIME_MAX_SESSIONS, value: "1"}
            - {name: REALTIME_UI_CONFIG_JSON, value: '{"targetFps":${REALTIME_TARGET_FPS},"size":"${REALTIME_SIZE}","dualModels":{"minwm":{"sinkSize":8,"windowFrames":32}}}'}
          ports:
            - {name: denoiser-1, containerPort: 30010}
          startupProbe:
            httpGet: {path: /health, port: denoiser-1}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 360
          readinessProbe:
            httpGet: {path: /health, port: denoiser-1}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          livenessProbe:
            httpGet: {path: /health, port: denoiser-1}
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 6
          resources:
            requests: {cpu: "16", memory: 120Gi}
            limits: {cpu: "24", memory: 180Gi}
          securityContext:
            capabilities:
              add: [SYS_ADMIN]
          volumeMounts:
            - {name: worker-epoch-denoiser-1, mountPath: /var/run/minwm-worker}
            - {name: startup-lock, mountPath: /var/run/minwm-startup-lock}
            - {name: model, mountPath: /models/minwm-tianpeng-gap12, readOnly: true}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /sgl-workspace/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: shm, mountPath: /dev/shm}
        - name: denoiser-2
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          workingDir: /opt/sglang
          command: [/bin/bash, -lc]
          args:
            - |
              set -euo pipefail
              exec 9>/var/run/minwm-startup-lock/denoiser.lock
              flock -x 9
              python3 -m sglang.multimodal_gen.tools.minwm_profile_launcher \
                --profile auto \
                --taehv-checkpoint-path /models/taehv/taew2_2.pth \
                -- \
                --model-path /models/minwm-tianpeng-gap12 \
                --num-gpus 1 \
                --tp-size 1 \
                --sp-degree 1 \
                --ulysses-degree 1 \
                --ring-degree 1 \
                --enable-cuda-graph true \
                --batching-max-size 1 \
                --batching-delay-ms 2 \
                --realtime-max-sessions 1 \
                --realtime-max-sessions-per-worker 1 \
                --realtime-vae-backend taehv_remote \
                --realtime-vae-transport websocket \
                --realtime-session-idle-timeout-s 90 \
                --realtime-session-max-lifetime-s 70 \
                --realtime-admission-wait-s 10 \
                --host 0.0.0.0 \
                --port 30020 \
                --master-port 30220 \
                --scheduler-port 5635 \
                --strict-ports &
              child=$!
              terminate() {
                kill -TERM "${child}" 2>/dev/null || true
                wait "${child}" || true
                exit 143
              }
              trap terminate TERM INT
              until curl --fail --silent --max-time 2 http://127.0.0.1:30020/health >/dev/null; do
                if ! kill -0 "${child}" 2>/dev/null; then
                  wait "${child}"
                  exit $?
                fi
                sleep 2
              done
              flock -u 9
              wait "${child}"
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: NVIDIA_VISIBLE_DEVICES, value: "2"}
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
            - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
            - {name: SGLANG_DISABLE_PDEATHSIG, value: "1"}
            - {name: OMP_NUM_THREADS, value: "4"}
            - {name: MKL_NUM_THREADS, value: "4"}
            - {name: OPENBLAS_NUM_THREADS, value: "4"}
            - {name: NUMEXPR_NUM_THREADS, value: "4"}
            - {name: VECLIB_MAXIMUM_THREADS, value: "4"}
            - {name: TOKENIZERS_PARALLELISM, value: "false"}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
            - {name: MINWM_ATTENTION_IMPL, value: packed}
            - {name: MINWM_SEGMENT_COMPILE, value: "true"}
            - {name: MINWM_CACHE_ROTATED_K, value: "true"}
            - {name: MINWM_PRECOMPUTE_CACHE_ROPE, value: "true"}
            - {name: MINWM_CACHE_PACKED_METADATA, value: "true"}
            - {name: MINWM_PACKED_ATTENTION_DETERMINISTIC, value: "false"}
            - {name: MINWM_NATIVE_COMPONENTS, value: ""}
            - {name: MINWM_RUNTIME_ALIGNMENT_LOG, value: "1"}
            - {name: SGLANG_MINWM_REQUIRE_SM120_FA4, value: "1"}
            - {name: SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D, value: "false"}
            - {name: NCCL_DEBUG, value: WARN}
            - {name: NCCL_PROTO, value: Simple}
            - {name: REALTIME_MAX_SESSIONS, value: "1"}
            - {name: REALTIME_UI_CONFIG_JSON, value: '{"targetFps":${REALTIME_TARGET_FPS},"size":"${REALTIME_SIZE}","dualModels":{"minwm":{"sinkSize":8,"windowFrames":32}}}'}
          ports:
            - {name: denoiser-2, containerPort: 30020}
          startupProbe:
            httpGet: {path: /health, port: denoiser-2}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 360
          readinessProbe:
            httpGet: {path: /health, port: denoiser-2}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          livenessProbe:
            httpGet: {path: /health, port: denoiser-2}
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 6
          resources:
            requests: {cpu: "16", memory: 120Gi}
            limits: {cpu: "24", memory: 180Gi}
          securityContext:
            capabilities:
              add: [SYS_ADMIN]
          volumeMounts:
            - {name: worker-epoch-denoiser-2, mountPath: /var/run/minwm-worker}
            - {name: startup-lock, mountPath: /var/run/minwm-startup-lock}
            - {name: model, mountPath: /models/minwm-tianpeng-gap12, readOnly: true}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /sgl-workspace/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: shm, mountPath: /dev/shm}
        - name: denoiser-3
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          workingDir: /opt/sglang
          command: [/bin/bash, -lc]
          args:
            - |
              set -euo pipefail
              exec 9>/var/run/minwm-startup-lock/denoiser.lock
              flock -x 9
              python3 -m sglang.multimodal_gen.tools.minwm_profile_launcher \
                --profile auto \
                --taehv-checkpoint-path /models/taehv/taew2_2.pth \
                -- \
                --model-path /models/minwm-tianpeng-gap12 \
                --num-gpus 1 \
                --tp-size 1 \
                --sp-degree 1 \
                --ulysses-degree 1 \
                --ring-degree 1 \
                --enable-cuda-graph true \
                --batching-max-size 1 \
                --batching-delay-ms 2 \
                --realtime-max-sessions 1 \
                --realtime-max-sessions-per-worker 1 \
                --realtime-vae-backend taehv_remote \
                --realtime-vae-transport websocket \
                --realtime-session-idle-timeout-s 90 \
                --realtime-session-max-lifetime-s 70 \
                --realtime-admission-wait-s 10 \
                --host 0.0.0.0 \
                --port 30030 \
                --master-port 30230 \
                --scheduler-port 5645 \
                --strict-ports &
              child=$!
              terminate() {
                kill -TERM "${child}" 2>/dev/null || true
                wait "${child}" || true
                exit 143
              }
              trap terminate TERM INT
              until curl --fail --silent --max-time 2 http://127.0.0.1:30030/health >/dev/null; do
                if ! kill -0 "${child}" 2>/dev/null; then
                  wait "${child}"
                  exit $?
                fi
                sleep 2
              done
              flock -u 9
              wait "${child}"
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: NVIDIA_VISIBLE_DEVICES, value: "3"}
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
            - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
            - {name: SGLANG_DISABLE_PDEATHSIG, value: "1"}
            - {name: OMP_NUM_THREADS, value: "4"}
            - {name: MKL_NUM_THREADS, value: "4"}
            - {name: OPENBLAS_NUM_THREADS, value: "4"}
            - {name: NUMEXPR_NUM_THREADS, value: "4"}
            - {name: VECLIB_MAXIMUM_THREADS, value: "4"}
            - {name: TOKENIZERS_PARALLELISM, value: "false"}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
            - {name: MINWM_ATTENTION_IMPL, value: packed}
            - {name: MINWM_SEGMENT_COMPILE, value: "true"}
            - {name: MINWM_CACHE_ROTATED_K, value: "true"}
            - {name: MINWM_PRECOMPUTE_CACHE_ROPE, value: "true"}
            - {name: MINWM_CACHE_PACKED_METADATA, value: "true"}
            - {name: MINWM_PACKED_ATTENTION_DETERMINISTIC, value: "false"}
            - {name: MINWM_NATIVE_COMPONENTS, value: ""}
            - {name: MINWM_RUNTIME_ALIGNMENT_LOG, value: "1"}
            - {name: SGLANG_MINWM_REQUIRE_SM120_FA4, value: "1"}
            - {name: SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D, value: "false"}
            - {name: NCCL_DEBUG, value: WARN}
            - {name: NCCL_PROTO, value: Simple}
            - {name: REALTIME_MAX_SESSIONS, value: "1"}
            - {name: REALTIME_UI_CONFIG_JSON, value: '{"targetFps":${REALTIME_TARGET_FPS},"size":"${REALTIME_SIZE}","dualModels":{"minwm":{"sinkSize":8,"windowFrames":32}}}'}
          ports:
            - {name: denoiser-3, containerPort: 30030}
          startupProbe:
            httpGet: {path: /health, port: denoiser-3}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 360
          readinessProbe:
            httpGet: {path: /health, port: denoiser-3}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          livenessProbe:
            httpGet: {path: /health, port: denoiser-3}
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 6
          resources:
            requests: {cpu: "16", memory: 120Gi}
            limits: {cpu: "24", memory: 180Gi}
          securityContext:
            capabilities:
              add: [SYS_ADMIN]
          volumeMounts:
            - {name: worker-epoch-denoiser-3, mountPath: /var/run/minwm-worker}
            - {name: startup-lock, mountPath: /var/run/minwm-startup-lock}
            - {name: model, mountPath: /models/minwm-tianpeng-gap12, readOnly: true}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /sgl-workspace/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: shm, mountPath: /dev/shm}
        - name: denoiser-4
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          workingDir: /opt/sglang
          command: [/bin/bash, -lc]
          args:
            - |
              set -euo pipefail
              exec 9>/var/run/minwm-startup-lock/denoiser.lock
              flock -x 9
              python3 -m sglang.multimodal_gen.tools.minwm_profile_launcher \
                --profile auto \
                --taehv-checkpoint-path /models/taehv/taew2_2.pth \
                -- \
                --model-path /models/minwm-tianpeng-gap12 \
                --num-gpus 1 \
                --tp-size 1 \
                --sp-degree 1 \
                --ulysses-degree 1 \
                --ring-degree 1 \
                --enable-cuda-graph true \
                --batching-max-size 1 \
                --batching-delay-ms 2 \
                --realtime-max-sessions 1 \
                --realtime-max-sessions-per-worker 1 \
                --realtime-vae-backend taehv_remote \
                --realtime-vae-transport websocket \
                --realtime-session-idle-timeout-s 90 \
                --realtime-session-max-lifetime-s 70 \
                --realtime-admission-wait-s 10 \
                --host 0.0.0.0 \
                --port 30040 \
                --master-port 30240 \
                --scheduler-port 5655 \
                --strict-ports &
              child=$!
              terminate() {
                kill -TERM "${child}" 2>/dev/null || true
                wait "${child}" || true
                exit 143
              }
              trap terminate TERM INT
              until curl --fail --silent --max-time 2 http://127.0.0.1:30040/health >/dev/null; do
                if ! kill -0 "${child}" 2>/dev/null; then
                  wait "${child}"
                  exit $?
                fi
                sleep 2
              done
              flock -u 9
              wait "${child}"
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: NVIDIA_VISIBLE_DEVICES, value: "4"}
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
            - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
            - {name: SGLANG_DISABLE_PDEATHSIG, value: "1"}
            - {name: OMP_NUM_THREADS, value: "4"}
            - {name: MKL_NUM_THREADS, value: "4"}
            - {name: OPENBLAS_NUM_THREADS, value: "4"}
            - {name: NUMEXPR_NUM_THREADS, value: "4"}
            - {name: VECLIB_MAXIMUM_THREADS, value: "4"}
            - {name: TOKENIZERS_PARALLELISM, value: "false"}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
            - {name: MINWM_ATTENTION_IMPL, value: packed}
            - {name: MINWM_SEGMENT_COMPILE, value: "true"}
            - {name: MINWM_CACHE_ROTATED_K, value: "true"}
            - {name: MINWM_PRECOMPUTE_CACHE_ROPE, value: "true"}
            - {name: MINWM_CACHE_PACKED_METADATA, value: "true"}
            - {name: MINWM_PACKED_ATTENTION_DETERMINISTIC, value: "false"}
            - {name: MINWM_NATIVE_COMPONENTS, value: ""}
            - {name: MINWM_RUNTIME_ALIGNMENT_LOG, value: "1"}
            - {name: SGLANG_MINWM_REQUIRE_SM120_FA4, value: "1"}
            - {name: SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D, value: "false"}
            - {name: NCCL_DEBUG, value: WARN}
            - {name: NCCL_PROTO, value: Simple}
            - {name: REALTIME_MAX_SESSIONS, value: "1"}
            - {name: REALTIME_UI_CONFIG_JSON, value: '{"targetFps":${REALTIME_TARGET_FPS},"size":"${REALTIME_SIZE}","dualModels":{"minwm":{"sinkSize":8,"windowFrames":32}}}'}
          ports:
            - {name: denoiser-4, containerPort: 30040}
          startupProbe:
            httpGet: {path: /health, port: denoiser-4}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 360
          readinessProbe:
            httpGet: {path: /health, port: denoiser-4}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          livenessProbe:
            httpGet: {path: /health, port: denoiser-4}
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 6
          resources:
            requests: {cpu: "16", memory: 120Gi}
            limits: {cpu: "24", memory: 180Gi}
          securityContext:
            capabilities:
              add: [SYS_ADMIN]
          volumeMounts:
            - {name: worker-epoch-denoiser-4, mountPath: /var/run/minwm-worker}
            - {name: startup-lock, mountPath: /var/run/minwm-startup-lock}
            - {name: model, mountPath: /models/minwm-tianpeng-gap12, readOnly: true}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /sgl-workspace/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: shm, mountPath: /dev/shm}
        - name: denoiser-5
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          workingDir: /opt/sglang
          command: [/bin/bash, -lc]
          args:
            - |
              set -euo pipefail
              exec 9>/var/run/minwm-startup-lock/denoiser.lock
              flock -x 9
              python3 -m sglang.multimodal_gen.tools.minwm_profile_launcher \
                --profile auto \
                --taehv-checkpoint-path /models/taehv/taew2_2.pth \
                -- \
                --model-path /models/minwm-tianpeng-gap12 \
                --num-gpus 1 \
                --tp-size 1 \
                --sp-degree 1 \
                --ulysses-degree 1 \
                --ring-degree 1 \
                --enable-cuda-graph true \
                --batching-max-size 1 \
                --batching-delay-ms 2 \
                --realtime-max-sessions 1 \
                --realtime-max-sessions-per-worker 1 \
                --realtime-vae-backend taehv_remote \
                --realtime-vae-transport websocket \
                --realtime-session-idle-timeout-s 90 \
                --realtime-session-max-lifetime-s 70 \
                --realtime-admission-wait-s 10 \
                --host 0.0.0.0 \
                --port 30050 \
                --master-port 30250 \
                --scheduler-port 5665 \
                --strict-ports &
              child=$!
              terminate() {
                kill -TERM "${child}" 2>/dev/null || true
                wait "${child}" || true
                exit 143
              }
              trap terminate TERM INT
              until curl --fail --silent --max-time 2 http://127.0.0.1:30050/health >/dev/null; do
                if ! kill -0 "${child}" 2>/dev/null; then
                  wait "${child}"
                  exit $?
                fi
                sleep 2
              done
              flock -u 9
              wait "${child}"
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: NVIDIA_VISIBLE_DEVICES, value: "5"}
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
            - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
            - {name: SGLANG_DISABLE_PDEATHSIG, value: "1"}
            - {name: OMP_NUM_THREADS, value: "4"}
            - {name: MKL_NUM_THREADS, value: "4"}
            - {name: OPENBLAS_NUM_THREADS, value: "4"}
            - {name: NUMEXPR_NUM_THREADS, value: "4"}
            - {name: VECLIB_MAXIMUM_THREADS, value: "4"}
            - {name: TOKENIZERS_PARALLELISM, value: "false"}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
            - {name: MINWM_ATTENTION_IMPL, value: packed}
            - {name: MINWM_SEGMENT_COMPILE, value: "true"}
            - {name: MINWM_CACHE_ROTATED_K, value: "true"}
            - {name: MINWM_PRECOMPUTE_CACHE_ROPE, value: "true"}
            - {name: MINWM_CACHE_PACKED_METADATA, value: "true"}
            - {name: MINWM_PACKED_ATTENTION_DETERMINISTIC, value: "false"}
            - {name: MINWM_NATIVE_COMPONENTS, value: ""}
            - {name: MINWM_RUNTIME_ALIGNMENT_LOG, value: "1"}
            - {name: SGLANG_MINWM_REQUIRE_SM120_FA4, value: "1"}
            - {name: SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D, value: "false"}
            - {name: NCCL_DEBUG, value: WARN}
            - {name: NCCL_PROTO, value: Simple}
            - {name: REALTIME_MAX_SESSIONS, value: "1"}
            - {name: REALTIME_UI_CONFIG_JSON, value: '{"targetFps":${REALTIME_TARGET_FPS},"size":"${REALTIME_SIZE}","dualModels":{"minwm":{"sinkSize":8,"windowFrames":32}}}'}
          ports:
            - {name: denoiser-5, containerPort: 30050}
          startupProbe:
            httpGet: {path: /health, port: denoiser-5}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 360
          readinessProbe:
            httpGet: {path: /health, port: denoiser-5}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          livenessProbe:
            httpGet: {path: /health, port: denoiser-5}
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 6
          resources:
            requests: {cpu: "16", memory: 120Gi}
            limits: {cpu: "24", memory: 180Gi}
          securityContext:
            capabilities:
              add: [SYS_ADMIN]
          volumeMounts:
            - {name: worker-epoch-denoiser-5, mountPath: /var/run/minwm-worker}
            - {name: startup-lock, mountPath: /var/run/minwm-startup-lock}
            - {name: model, mountPath: /models/minwm-tianpeng-gap12, readOnly: true}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /sgl-workspace/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: shm, mountPath: /dev/shm}
        - name: denoiser-6
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          workingDir: /opt/sglang
          command: [/bin/bash, -lc]
          args:
            - |
              set -euo pipefail
              exec 9>/var/run/minwm-startup-lock/denoiser.lock
              flock -x 9
              python3 -m sglang.multimodal_gen.tools.minwm_profile_launcher \
                --profile auto \
                --taehv-checkpoint-path /models/taehv/taew2_2.pth \
                -- \
                --model-path /models/minwm-tianpeng-gap12 \
                --num-gpus 1 \
                --tp-size 1 \
                --sp-degree 1 \
                --ulysses-degree 1 \
                --ring-degree 1 \
                --enable-cuda-graph true \
                --batching-max-size 1 \
                --batching-delay-ms 2 \
                --realtime-max-sessions 1 \
                --realtime-max-sessions-per-worker 1 \
                --realtime-vae-backend taehv_remote \
                --realtime-vae-transport websocket \
                --realtime-session-idle-timeout-s 90 \
                --realtime-session-max-lifetime-s 70 \
                --realtime-admission-wait-s 10 \
                --host 0.0.0.0 \
                --port 30060 \
                --master-port 30260 \
                --scheduler-port 5675 \
                --strict-ports &
              child=$!
              terminate() {
                kill -TERM "${child}" 2>/dev/null || true
                wait "${child}" || true
                exit 143
              }
              trap terminate TERM INT
              until curl --fail --silent --max-time 2 http://127.0.0.1:30060/health >/dev/null; do
                if ! kill -0 "${child}" 2>/dev/null; then
                  wait "${child}"
                  exit $?
                fi
                sleep 2
              done
              flock -u 9
              wait "${child}"
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python:/sgl-workspace/sglang/python}
            - {name: NVIDIA_VISIBLE_DEVICES, value: "6"}
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
            - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True"}
            - {name: SGLANG_DISABLE_PDEATHSIG, value: "1"}
            - {name: OMP_NUM_THREADS, value: "4"}
            - {name: MKL_NUM_THREADS, value: "4"}
            - {name: OPENBLAS_NUM_THREADS, value: "4"}
            - {name: NUMEXPR_NUM_THREADS, value: "4"}
            - {name: VECLIB_MAXIMUM_THREADS, value: "4"}
            - {name: TOKENIZERS_PARALLELISM, value: "false"}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
            - {name: MINWM_ATTENTION_IMPL, value: packed}
            - {name: MINWM_SEGMENT_COMPILE, value: "true"}
            - {name: MINWM_CACHE_ROTATED_K, value: "true"}
            - {name: MINWM_PRECOMPUTE_CACHE_ROPE, value: "true"}
            - {name: MINWM_CACHE_PACKED_METADATA, value: "true"}
            - {name: MINWM_PACKED_ATTENTION_DETERMINISTIC, value: "false"}
            - {name: MINWM_NATIVE_COMPONENTS, value: ""}
            - {name: MINWM_RUNTIME_ALIGNMENT_LOG, value: "1"}
            - {name: SGLANG_MINWM_REQUIRE_SM120_FA4, value: "1"}
            - {name: SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D, value: "false"}
            - {name: NCCL_DEBUG, value: WARN}
            - {name: NCCL_PROTO, value: Simple}
            - {name: REALTIME_MAX_SESSIONS, value: "1"}
            - {name: REALTIME_UI_CONFIG_JSON, value: '{"targetFps":${REALTIME_TARGET_FPS},"size":"${REALTIME_SIZE}","dualModels":{"minwm":{"sinkSize":8,"windowFrames":32}}}'}
          ports:
            - {name: denoiser-6, containerPort: 30060}
          startupProbe:
            httpGet: {path: /health, port: denoiser-6}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 360
          readinessProbe:
            httpGet: {path: /health, port: denoiser-6}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          livenessProbe:
            httpGet: {path: /health, port: denoiser-6}
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 6
          resources:
            requests: {cpu: "16", memory: 120Gi}
            limits: {cpu: "24", memory: 180Gi}
          securityContext:
            capabilities:
              add: [SYS_ADMIN]
          volumeMounts:
            - {name: worker-epoch-denoiser-6, mountPath: /var/run/minwm-worker}
            - {name: startup-lock, mountPath: /var/run/minwm-startup-lock}
            - {name: model, mountPath: /models/minwm-tianpeng-gap12, readOnly: true}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: runtime-tools-patch, mountPath: /sgl-workspace/sglang/python/sglang/multimodal_gen/tools/minwm_profile_launcher.py, subPath: minwm_profile_launcher.py, readOnly: true}
            - {name: shm, mountPath: /dev/shm}
      volumes:
        - {name: worker-epoch-denoiser-0, emptyDir: {}}
        - {name: worker-epoch-denoiser-1, emptyDir: {}}
        - {name: worker-epoch-denoiser-2, emptyDir: {}}
        - {name: worker-epoch-denoiser-3, emptyDir: {}}
        - {name: worker-epoch-denoiser-4, emptyDir: {}}
        - {name: worker-epoch-denoiser-5, emptyDir: {}}
        - {name: worker-epoch-denoiser-6, emptyDir: {}}
        - {name: startup-lock, emptyDir: {}}
        - {name: shm, emptyDir: {medium: Memory, sizeLimit: 256Gi}}
        - {name: runtime-tools-patch, configMap: {name: zing-runtime-tools-patch}}
        - name: model
          hostPath:
            path: /data/zing-realtime/model-cache/zing/model
            type: Directory
        - name: taehv
          hostPath:
            path: /data/zing-realtime/taehv
            type: Directory
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: zing-vae-5090-dual
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-vae-5090-dual
    app.kubernetes.io/part-of: minwm-realtime
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-vae-5090-dual
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
  template:
    metadata:
      annotations:
        seedleap.ai/runtime-source-patch: ${RUNTIME_SOURCE_PATCH_VERSION}
      labels:
        app.kubernetes.io/name: zing-vae-5090-dual
        app.kubernetes.io/part-of: minwm-realtime
        seedleap.ai/worker-role: vae
        seedleap.ai/gpu-topology: 7p1-sp1
    spec:
      serviceAccountName: zing-realtime
      runtimeClassName: nvidia
      terminationGracePeriodSeconds: 90
      nodeSelector:
        seedleap.ai/gpu-pool: aliyun-beijing-5090
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      initContainers:
        - name: vae-heartbeat-0
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:18082/health
              --state-url=http://127.0.0.1:18082/v1/realtime_worker/state
              --worker-id=$(NODE_NAME)-vae-0
              --worker-epoch-file=/var/run/minwm-worker/epoch
              --role=vae
              --endpoint=ws://$(POD_IP):18082/v1/realtime_vae/decode
              --reservation-endpoint=http://$(POD_IP):18082/v1/realtime_worker
              --node-name=$(NODE_NAME)
              --capacity=16
              --model-revision=all
              --vae-fingerprint=taew2_2-d053e216
              --interval-s=5
          env:
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: void}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          resources:
            requests: {cpu: 25m, memory: 64Mi}
            limits: {cpu: 200m, memory: 256Mi}
          volumeMounts:
            - {name: worker-epoch-vae-0, mountPath: /var/run/minwm-worker}
      containers:
        - name: vae-0
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [python3]
          args:
            - -m
            - sglang.multimodal_gen.runtime.entrypoints.realtime_vae_server
            - --decoder-backend=taehv
            - --checkpoint-path=/models/taehv/taew2_2.pth
            - --device=cuda
            - --dtype=bfloat16
            - --max-sessions=16
            - --queue-depth-per-session=1
            - --encoded-frames-per-batch=1
            - --max-message-mb=64
            - --host=0.0.0.0
            - --port=18082
            - --direct-h264-output
            - --direct-h264-trigger-output-format=jpeg
            - --h264-ffmpeg-bin=${H264_FFMPEG_BIN}
            - --h264-fps=${REALTIME_TARGET_FPS}
            - --h264-bitrate-kbps=3000
            - --h264-crf=20
            - --h264-preset=fast
            - --h264-gop-seconds=2
            - --h264-vbv-buffer-ms=250
            - --h264-max-frame-age-ms=250
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - {name: NVIDIA_VISIBLE_DEVICES, value: "7"}
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: "compute,utility"}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
          ports:
            - {name: vae-0, containerPort: 18082}
          startupProbe:
            httpGet: {path: /health, port: vae-0}
            periodSeconds: 5
            timeoutSeconds: 5
            failureThreshold: 120
          readinessProbe:
            httpGet: {path: /health, port: vae-0}
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 6
          livenessProbe:
            httpGet: {path: /health, port: vae-0}
            initialDelaySeconds: 30
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 6
          resources:
            requests: {cpu: "6", memory: 24Gi}
            limits: {cpu: "8", memory: 32Gi}
          volumeMounts:
            - {name: worker-epoch-vae-0, mountPath: /var/run/minwm-worker}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
            - {name: runtime-realtime-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/realtime, readOnly: true}
            - {name: runtime-entrypoint-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/entrypoints/realtime_vae_server.py, subPath: realtime_vae_server.py, readOnly: true}
            - {name: runtime-utils-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/runtime/utils, readOnly: true}
            - {name: runtime-envs-patch, mountPath: /opt/sglang/python/sglang/multimodal_gen/envs.py, subPath: envs.py, readOnly: true}
            - {name: runtime-dep-shims, mountPath: /opt/sglang/python/prometheus_client.py, subPath: prometheus_client.py, readOnly: true}
            - {name: runtime-dep-shims, mountPath: /opt/sglang/python/sitecustomize.py, subPath: sitecustomize.py, readOnly: true}
      volumes:
        - {name: worker-epoch-vae-0, emptyDir: {}}
        - {name: runtime-realtime-patch, configMap: {name: zing-runtime-realtime-patch}}
        - {name: runtime-entrypoint-patch, configMap: {name: zing-runtime-entrypoint-patch}}
        - {name: runtime-utils-patch, configMap: {name: zing-runtime-utils-patch}}
        - {name: runtime-envs-patch, configMap: {name: zing-runtime-envs-patch}}
        - {name: runtime-dep-shims, configMap: {name: zing-runtime-dep-shims}}
        - name: taehv
          hostPath:
            path: /data/zing-realtime/taehv
            type: Directory
