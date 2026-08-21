apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
  labels:
    app.kubernetes.io/part-of: minwm-realtime
    seedleap.ai/environment: aliyun-beijing-pre-gpu
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: zing-realtime
  namespace: ${NAMESPACE}
imagePullSecrets:
  - name: acr-pull
---
apiVersion: v1
kind: Service
metadata:
  name: zing-coordinator
  namespace: ${NAMESPACE}
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: zing-coordinator
  ports:
    - name: http
      port: 18081
      targetPort: http
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zing-coordinator
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-coordinator
    app.kubernetes.io/part-of: minwm-realtime
spec:
  replicas: 1
  revisionHistoryLimit: 3
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-coordinator
  template:
    metadata:
      labels:
        app.kubernetes.io/name: zing-coordinator
        app.kubernetes.io/part-of: minwm-realtime
    spec:
      serviceAccountName: zing-realtime
      terminationGracePeriodSeconds: 30
      containers:
        - name: coordinator
          image: ${CONTROL_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_coordinator_server
              --host=0.0.0.0
              --port=18081
              --backend=memory
              --ttl-s=30
              --worker-ttl-s=15
              --wait-timeout-s=10
              --candidate-limit=64
              --denoiser-capacity-limit=7
              --vae-capacity-limit=16
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: OTEL_SERVICE_NAME, value: zing-coordinator}
          ports:
            - {name: http, containerPort: 18081}
          startupProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 2
            failureThreshold: 60
          readinessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 5
          livenessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 15
          resources:
            requests: {cpu: 250m, memory: 256Mi}
            limits: {cpu: "2", memory: 2Gi}
---
apiVersion: v1
kind: Service
metadata:
  name: zing-gateway
  namespace: ${NAMESPACE}
spec:
  type: ClusterIP
  selector:
    app.kubernetes.io/name: zing-gateway
  ports:
    - name: http
      port: 18080
      targetPort: http
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zing-gateway
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-gateway
    app.kubernetes.io/part-of: minwm-realtime
spec:
  replicas: 1
  revisionHistoryLimit: 3
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-gateway
  template:
    metadata:
      labels:
        app.kubernetes.io/name: zing-gateway
        app.kubernetes.io/part-of: minwm-realtime
    spec:
      serviceAccountName: zing-realtime
      terminationGracePeriodSeconds: 30
      containers:
        - name: gateway
          image: ${CONTROL_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [/bin/sh, -ec]
          args:
            - >-
              exec python3 -m sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server
              --host=0.0.0.0
              --port=18080
              --coordinator-url=http://zing-coordinator:18081
              --model-revision=wan22-5b-stage3-dmd-47-0808-2fb2cfec2a2
              --vae-fingerprint=taew2_2-d053e216
              --internal-output-url=ws://${POD_IP}:18080/v1/internal/realtime_output
              --output-queue-depth=64
              --output-enqueue-timeout-s=0
              --output-drain-timeout-s=70
              --lease-renew-interval-s=10
              --release-grace-s=0.5
              --max-admission-waiters=64
              --ui-config-json="$RUNTIME_UI_CONFIG_JSON"
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: OTEL_SERVICE_NAME, value: zing-gateway}
            - name: POD_IP
              valueFrom:
                fieldRef:
                  fieldPath: status.podIP
            - name: RUNTIME_UI_CONFIG_JSON
              value: '${UI_CONFIG_JSON}'
          ports:
            - name: http
              containerPort: 18080
              hostPort: 18080
          startupProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 2
            failureThreshold: 60
          readinessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 5
          livenessProbe:
            httpGet: {path: /healthz, port: http}
            periodSeconds: 15
          resources:
            requests: {cpu: 500m, memory: 512Mi}
            limits: {cpu: "4", memory: 4Gi}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: zing-webui
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: zing-webui
    app.kubernetes.io/part-of: minwm-realtime
spec:
  replicas: 1
  revisionHistoryLimit: 3
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app.kubernetes.io/name: zing-webui
  template:
    metadata:
      labels:
        app.kubernetes.io/name: zing-webui
        app.kubernetes.io/part-of: minwm-realtime
    spec:
      serviceAccountName: zing-realtime
      automountServiceAccountToken: false
      terminationGracePeriodSeconds: 10
      securityContext:
        runAsNonRoot: true
        runAsUser: 65532
        runAsGroup: 65532
        fsGroup: 65532
        fsGroupChangePolicy: OnRootMismatch
      containers:
        - name: webui
          image: ${WEBUI_IMAGE}
          imagePullPolicy: IfNotPresent
          command: [python]
          args:
            - /opt/sglang/python/sglang/multimodal_gen/apps/realtime_webui/server.py
          env:
            - {name: PYTHONUNBUFFERED, value: "1"}
            - {name: PYTHONPATH, value: /opt/sglang/python}
            - {name: WEBUI_PORT, value: "18080"}
            - {name: REALTIME_UPSTREAM_HTTP, value: http://zing-gateway:18080}
            - {name: REALTIME_UPSTREAM_WS, value: ws://zing-gateway:18080}
            - {name: MINWM_UPSTREAM_HTTP, value: http://zing-gateway:18080/backends/minwm}
            - {name: MINWM_UPSTREAM_WS, value: ws://zing-gateway:18080/backends/minwm}
            - {name: VIDEO_PROMPT_REWRITE_PROVIDER, value: local}
            - {name: VIDEO_PROMPT_REWRITE_CREDENTIALS, value: /run/secrets/realtime-webui/prompt-rewriter-vertex.json}
            - name: REALTIME_UI_CONFIG_JSON
              value: '${UI_CONFIG_JSON}'
            - name: HTTPS_PROXY
              valueFrom:
                secretKeyRef: {name: zing-proxy-env, key: HTTPS_PROXY, optional: true}
            - name: HTTP_PROXY
              valueFrom:
                secretKeyRef: {name: zing-proxy-env, key: HTTP_PROXY, optional: true}
            - name: https_proxy
              valueFrom:
                secretKeyRef: {name: zing-proxy-env, key: https_proxy, optional: true}
            - name: http_proxy
              valueFrom:
                secretKeyRef: {name: zing-proxy-env, key: http_proxy, optional: true}
            - {name: WORLD_MODEL_METRIC_SERVICE, value: world-studio-webui}
            - {name: WORLD_MODEL_METRIC_LANE, value: aliyun-beijing}
          ports:
            - name: http
              containerPort: 18080
              hostPort: 80
          startupProbe:
            httpGet: {path: /runtime-config.js, port: http}
            periodSeconds: 2
            failureThreshold: 60
          readinessProbe:
            httpGet: {path: /runtime-config.js, port: http}
            periodSeconds: 5
          livenessProbe:
            httpGet: {path: /, port: http}
            periodSeconds: 15
          resources:
            requests: {cpu: 250m, memory: 384Mi}
            limits: {cpu: "2", memory: 2Gi}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: {drop: [ALL]}
          volumeMounts:
            - {name: webui-secrets, mountPath: /run/secrets/realtime-webui, readOnly: true}
            - {name: generated-images, mountPath: /opt/sglang/python/sglang/multimodal_gen/apps/realtime_webui_generated}
            - {name: tmp, mountPath: /tmp}
      volumes:
        - name: webui-secrets
          secret:
            secretName: webui-secrets
            optional: true
        - {name: generated-images, emptyDir: {sizeLimit: 1Gi}}
        - {name: tmp, emptyDir: {sizeLimit: 512Mi}}
