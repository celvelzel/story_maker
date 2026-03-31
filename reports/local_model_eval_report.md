# Local Model Evaluation Report

**Model:** merged_model_Qwen3-4B-Instruct-2507
**Base URL:** http://localhost:8000/v1
**Device:** NVIDIA GeForce RTX 3080 Ti (12GB VRAM)
**Deployment:** Local CUDA inference server

---

## ⏱️ Performance

- **Story generation latency:** 53.11s
- **JSON generation latency:** 9.06s
- **Average turn latency:** 9.05s

## 📊 Quality Metrics

- **Distinct-1:** 0.7555
- **Distinct-2:** 0.9956
- **Type-Token Ratio:** 0.7186
- **Flesch Reading Ease:** 72.57

## ✅ Functionality

- **JSON parsing:** FAIL - Model generates multiple JSON objects instead of one

## 📖 Sample Generated Story

```
1. The forest floor is littered with broken glass and rusted metal, the remnants of a skirmish between your forces and a local militia. Your squad leader, Captain Vex, stands at the edge of the clearing, his expression unreadable as he scans the horizon.

2. A sudden explosion rips through the underbrush ahead—then silence. Smoke curls from where the militia's artillery unit detonated its final charge. But the smoke doesn't rise straight up; it spirals wildly, forming shapes that shift like living things.

3. Your team regroups at the treeline, weapons drawn but not yet fired. The air smells of ozone and scorched earth. Someone whispers, "That wasn't a normal blast." Captain Vex nods slowly, his hand resting on the hilt of his sword. "We move deeper," he says quietly. "Whatever caused this is still out there."
```

## 🔍 Observations

1. **Latency:** Initial story generation takes ~53s (includes model warmup), subsequent requests ~9s
2. **Quality:** High distinct-2 score indicates good diversity, though responses can be verbose
3. **JSON Issue:** Model struggles with strict JSON formatting, often generating multiple objects
4. **GPU Usage:** ~8GB VRAM utilization
