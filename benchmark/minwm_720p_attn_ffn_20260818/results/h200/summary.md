# MinWM 720p attention / FFN Spot matrix

Reference: `packed-det-bf16`

Raw-speed ranking (the quality column is only a sampled corruption screen):

Ranking metric: `scheduler_fps`

| rank | lane | scheduler FPS | client FPS | speedup | peak MiB | sampled PSNR | screen |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 1 | `dense-fa-online-fp8` | 9.824 | 9.814 | +26.55% | 57069 | 9.54 | no |
| 2 | `dense-fa-static-ffn-fp8` | 9.732 | 9.722 | +25.36% | 59251 | 10.36 | no |
| 3 | `dense-fa-bf16` | 9.272 | 9.264 | +19.44% | 61803 | 10.98 | no |
| 4 | `packed-fast-static-ffn-fp8` | 8.100 | 8.093 | +4.35% | 59251 | 10.47 | no |
| 5 | `packed-fast-bf16` | 7.897 | 7.891 | +1.73% | 61801 | 9.99 | no |
| 6 | `packed-det-bf16` | 7.763 | 7.755 | +0.00% | 61809 | lossless | yes |
| 7 | `dense-sdpa-bf16` | 7.754 | 7.748 | -0.11% | 61801 | 9.82 | no |

Quality-screened order: `packed-det-bf16`

Failed or skipped lanes: `dense-cross-sage2-bf16`, `dense-fa-bf16-compile`, `dense-fa-static-ffn-fp8-compile`, `packed-fast-static-ffn-fp8-compile`
