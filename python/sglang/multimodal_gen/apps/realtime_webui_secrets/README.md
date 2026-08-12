# Live Direction prompt rewriter secret

The server reads a Vertex service-account credential from this project-local
path by default:

`prompt-rewriter-vertex.json`

This directory is deliberately outside the WebUI static-file root. The JSON
credential is ignored by Git and is never served to the browser. In a
deployment, mount the Kubernetes Secret at the same path, or set
`VIDEO_PROMPT_REWRITE_CREDENTIALS` to another in-container path.

Optional server-side settings:

- `VIDEO_PROMPT_REWRITE_PROJECT_ID`
- `VIDEO_PROMPT_REWRITE_MODEL` (default: `gemini-3.1-flash-lite`)
- `VIDEO_PROMPT_REWRITE_VERTEX_LOCATION` (default: `global`)
- `VIDEO_PROMPT_REWRITE_TIMEOUT_SECONDS` (default: `20`)
