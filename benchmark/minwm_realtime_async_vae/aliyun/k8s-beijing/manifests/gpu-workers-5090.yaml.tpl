apiVersion: apps/v1
kind: Deployment
metadata:
  name: zing-vae-5090
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-vae-5090
    app.kubernetes.io/part-of: minwm-realtime
spec:
  replicas: 0
  revisionHistoryLimit: 2
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-vae-5090
  template:
    metadata:
      labels:
        app.kubernetes.io/name: zing-vae-5090
        app.kubernetes.io/part-of: minwm-realtime
        seedleap.ai/worker-role: vae
    spec:
      serviceAccountName: zing-realtime
      nodeSelector:
        seedleap.ai/gpu-pool: aliyun-beijing-5090
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      initContainers:
        - name: vae-heartbeat
          image: ${CONTROL_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:18082/health
              --state-url=http://127.0.0.1:18082/v1/realtime_worker/state
              --worker-id=$(POD_UID)
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
            - name: POD_UID
              valueFrom: {fieldRef: {fieldPath: metadata.uid}}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          volumeMounts:
            - {name: worker-epoch, mountPath: /var/run/minwm-worker}
      containers:
        - name: vae
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
            - --encode-workers=4
            - --direct-h264-output
            - --direct-h264-trigger-output-format=jpeg
            - --h264-fps=24
            - --h264-threads=2
            - --h264-preset=fast
            - --h264-profile=main
            - --h264-crf=20
            - --h264-bitrate-kbps=3000
            - --h264-vbv-buffer-ms=250
            - --h264-gop-seconds=2
            - --h264-max-queued-frames=24
            - --h264-max-frame-age-ms=250
            - --h264-live-edge-frames=6
            - --h264-startup-drop-frames=0
            - --max-message-mb=64
            - --host=0.0.0.0
            - --port=18082
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
          ports:
            - {name: api, containerPort: 18082}
          resources:
            requests: {cpu: "6", memory: 24Gi, nvidia.com/gpu: "1"}
            limits: {cpu: "8", memory: 32Gi, nvidia.com/gpu: "1"}
          volumeMounts:
            - {name: worker-epoch, mountPath: /var/run/minwm-worker}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
      volumes:
        - {name: worker-epoch, emptyDir: {}}
        - name: taehv
          hostPath:
            path: /data/zing-realtime/taehv
            type: Directory
---
apiVersion: v1
kind: Service
metadata:
  name: zing-denoiser-5090
  namespace: ${NAMESPACE}
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector:
    app.kubernetes.io/name: zing-denoiser-5090
  ports:
    - {name: api, port: 30000, targetPort: api}
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: zing-denoiser-5090
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-denoiser-5090
    app.kubernetes.io/part-of: minwm-realtime
spec:
  replicas: 0
  revisionHistoryLimit: 2
  serviceName: zing-denoiser-5090
  podManagementPolicy: Parallel
  updateStrategy:
    type: OnDelete
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-denoiser-5090
  template:
    metadata:
      labels:
        app.kubernetes.io/name: zing-denoiser-5090
        app.kubernetes.io/part-of: minwm-realtime
        seedleap.ai/worker-role: denoiser
    spec:
      serviceAccountName: zing-realtime
      terminationGracePeriodSeconds: 90
      nodeSelector:
        seedleap.ai/gpu-pool: aliyun-beijing-5090
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      initContainers:
        - name: denoiser-heartbeat
          image: ${CONTROL_IMAGE}
          imagePullPolicy: IfNotPresent
          restartPolicy: Always
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_worker_heartbeat
              --coordinator-url=http://zing-coordinator:18081
              --health-url=http://127.0.0.1:30000/health
              --state-url=http://127.0.0.1:30000/v1/realtime_worker/state
              --worker-id=$(POD_UID)
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
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: SGLANG_LIGHTWEIGHT_RUNTIME, value: "1"}
            - name: POD_UID
              valueFrom: {fieldRef: {fieldPath: metadata.uid}}
            - name: POD_IP
              valueFrom: {fieldRef: {fieldPath: status.podIP}}
            - name: NODE_NAME
              valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
          volumeMounts:
            - {name: worker-epoch, mountPath: /var/run/minwm-worker}
      containers:
        - name: denoiser
          image: ${GPU_RUNTIME_IMAGE}
          imagePullPolicy: IfNotPresent
          workingDir: /opt/sglang
          command: [/bin/bash, -lc]
          args:
            - |
              set -euo pipefail
              exec python3 -m sglang.multimodal_gen.tools.minwm_profile_launcher \
                --profile auto \
                --taehv-checkpoint-path /models/taehv/taew2_2.pth \
                -- \
                --model-path /models/minwm-tianpeng-gap12 \
                --batching-max-size 1 \
                --batching-delay-ms 2 \
                --realtime-max-sessions 1 \
                --realtime-max-sessions-per-worker 1 \
                --realtime-vae-backend taehv_remote \
                --realtime-vae-transport websocket \
                --realtime-session-idle-timeout-s 90 \
                --realtime-session-max-lifetime-s 70 \
                --realtime-worker-max-consumed-age-s 120 \
                --realtime-admission-wait-s 10 \
                --host 0.0.0.0 \
                --port 30000 \
                --attention-backend fa
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: PYTORCH_CUDA_ALLOC_CONF, value: expandable_segments:True}
            - {name: SGLANG_DISABLE_PDEATHSIG, value: "1"}
            - {name: OMP_NUM_THREADS, value: "4"}
            - {name: MKL_NUM_THREADS, value: "4"}
            - {name: OPENBLAS_NUM_THREADS, value: "4"}
            - {name: NUMEXPR_NUM_THREADS, value: "4"}
            - {name: VECLIB_MAXIMUM_THREADS, value: "4"}
            - {name: TOKENIZERS_PARALLELISM, value: "false"}
            - {name: WORKER_EPOCH_FILE, value: /var/run/minwm-worker/epoch}
            - {name: MINWM_ATTENTION_IMPL, value: packed}
            - {name: MINWM_CACHE_ROTATED_K, value: "false"}
            - {name: MINWM_PACKED_ATTENTION_DETERMINISTIC, value: "false"}
            - {name: MINWM_NATIVE_COMPONENTS, value: ""}
            - {name: SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D, value: "false"}
            - {name: NCCL_DEBUG, value: WARN}
            - {name: NCCL_PROTO, value: Simple}
            - {name: REALTIME_MAX_SESSIONS, value: "1"}
            - {name: REALTIME_UI_CONFIG_JSON, value: '{"targetFps":24,"size":"832x480","dualModels":{"minwm":{"sinkSize":8,"windowFrames":32}}}'}
          ports:
            - {name: api, containerPort: 30000}
          resources:
            requests: {cpu: "24", memory: 300Gi, nvidia.com/gpu: "1"}
            limits: {cpu: "48", memory: 400Gi, nvidia.com/gpu: "1"}
          volumeMounts:
            - {name: worker-epoch, mountPath: /var/run/minwm-worker}
            - {name: model, mountPath: /models/minwm-tianpeng-gap12, readOnly: true}
            - {name: taehv, mountPath: /models/taehv, readOnly: true}
      volumes:
        - {name: worker-epoch, emptyDir: {}}
        - name: model
          hostPath:
            path: /data/zing-realtime/model-cache/zing/model
            type: Directory
        - name: taehv
          hostPath:
            path: /data/zing-realtime/taehv
            type: Directory
