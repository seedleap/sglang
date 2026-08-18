# MinWM 720p attention / FFN Spot matrix

Performance reference: `packed-fast-bf16`

Quality byte reference: `packed-det-bf16`

Raw-speed ranking (the quality column is only a sampled corruption screen):

Ranking metric: `scheduler_fps`

| rank | lane | scheduler FPS | client FPS | speedup | peak MiB | sampled PSNR | screen |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 1 | `dense-fa-static-ffn-fp8` | 14.841 | 14.821 | +3.09% | 59410 | 11.15 | no |
| 2 | `dense-fa-online-fp8` | 14.762 | 14.741 | +2.54% | 57118 | 9.97 | no |
| 3 | `packed-fast-static-ffn-fp8` | 14.602 | 14.581 | +1.43% | 59410 | 9.85 | no |
| 4 | `dense-fa-bf16` | 14.452 | 14.430 | +0.39% | 61964 | 12.52 | no |
| 5 | `packed-fast-bf16` | 14.396 | 14.377 | +0.00% | 61962 | 11.54 | no |
| 6 | `packed-det-bf16` | 13.994 | 13.972 | -2.79% | 61968 | lossless | yes |
| 7 | `dense-sdpa-bf16` | 9.955 | 9.946 | -30.85% | 61964 | 11.66 | no |

Quality-screened order: `packed-det-bf16`

Failed or skipped lanes: `dense-cross-sage3-bf16`, `dense-fa-bf16-compile`, `dense-fa-static-ffn-fp8-compile`, `packed-fast-static-ffn-fp8-compile`
