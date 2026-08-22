apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
  labels:
    app.kubernetes.io/name: nvidia-device-plugin
    app.kubernetes.io/part-of: minwm-realtime
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: nvidia-device-plugin
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
  template:
    metadata:
      labels:
        app.kubernetes.io/name: nvidia-device-plugin
        app.kubernetes.io/part-of: minwm-realtime
    spec:
      runtimeClassName: nvidia
      priorityClassName: system-node-critical
      nodeSelector:
        seedleap.ai/gpu-worker: "true"
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
        - operator: Exists
      containers:
        - name: nvidia-device-plugin-ctr
          image: ${NVIDIA_DEVICE_PLUGIN_IMAGE}
          imagePullPolicy: IfNotPresent
          args:
            - --fail-on-init-error=false
          env:
            - {name: NVIDIA_VISIBLE_DEVICES, value: all}
            - {name: NVIDIA_DRIVER_CAPABILITIES, value: compute,utility}
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop: [ALL]
          volumeMounts:
            - name: device-plugin
              mountPath: /var/lib/kubelet/device-plugins
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins
            type: Directory
