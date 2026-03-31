#!/usr/bin/env python3
"""Simple direct evaluation of the local model deployment."""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.api_client import llm_client
from src.evaluation.metrics import distinct_n, self_bleu, type_token_ratio, flesch_reading_ease

def test_story_generation():
    """Test story generation quality."""
    print("=" * 60)
    print("TEST 1: Story Generation Quality")
    print("=" * 60)
    
    messages = [
        {"role": "system", "content": "You are an expert interactive-fiction narrator. Write in second person. Keep responses to 2-3 short paragraphs. Be concise."},
        {"role": "user", "content": "Create a brief opening scene for a fantasy adventure. Maximum 3 paragraphs."}
    ]
    
    start = time.time()
    response = llm_client.chat(messages, temperature=0.7, max_tokens=300)
    elapsed = time.time() - start
    
    print(f"\n⏱️  Latency: {elapsed:.2f}s")
    print(f"📝 Response length: {len(response)} chars, ~{len(response.split())} words")
    print(f"\n📖 Generated Story:\n{response[:500]}...")
    
    # Calculate metrics
    metrics = {
        "distinct_1": distinct_n([response], 1),
        "distinct_2": distinct_n([response], 2),
        "type_token_ratio": type_token_ratio(response),
        "flesch_reading_ease": flesch_reading_ease(response)
    }
    
    print(f"\n📊 Metrics:")
    for k, v in metrics.items():
        print(f"   - {k}: {v:.4f}")
    
    return response, elapsed, metrics

def test_option_generation():
    """Test JSON option generation."""
    print("\n" + "=" * 60)
    print("TEST 2: Option Generation (JSON)")
    print("=" * 60)
    
    story = "You stand at the entrance of a dark cave. The wind howls from within."
    
    messages = [
        {"role": "system", "content": "You generate JSON responses only. No prose."},
        {"role": "user", "content": f"""Generate exactly 2 player options as JSON.

Story: {story}

Return ONLY this JSON format, nothing else:
{{"options": [{{"text": "...", "risk_level": "low|medium|high"}}]}}"""}
    ]
    
    start = time.time()
    response = llm_client.chat(messages, temperature=0.5, max_tokens=200)
    elapsed = time.time() - start
    
    print(f"\n⏱️  Latency: {elapsed:.2f}s")
    print(f"\n📝 Raw Response:\n{response[:300]}")
    
    # Try to parse JSON
    try:
        # Find JSON in response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            print(f"\n✅ JSON parsed successfully!")
            print(f"   Options found: {len(parsed.get('options', []))}")
            for i, opt in enumerate(parsed.get('options', []), 1):
                print(f"   {i}. {opt.get('text', 'N/A')} (risk: {opt.get('risk_level', '?')})")
            return True, elapsed
        else:
            print(f"\n❌ No JSON found in response")
            return False, elapsed
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON parse error: {e}")
        return False, elapsed

def test_consistency():
    """Test multi-turn consistency."""
    print("\n" + "=" * 60)
    print("TEST 3: Multi-turn Consistency")
    print("=" * 60)
    
    latencies = []
    stories = []
    
    # Turn 1
    messages = [
        {"role": "system", "content": "You are a fantasy narrator. Write 1-2 short paragraphs only."},
        {"role": "user", "content": "You enter a tavern. Describe what you see in 1-2 short paragraphs."}
    ]
    
    start = time.time()
    resp1 = llm_client.chat(messages, temperature=0.7, max_tokens=200)
    latencies.append(time.time() - start)
    stories.append(resp1)
    print(f"\nTurn 1 ({latencies[0]:.2f}s): {resp1[:150]}...")
    
    # Turn 2
    messages.append({"role": "assistant", "content": resp1})
    messages.append({"role": "user", "content": "Talk to the bartender."})
    
    start = time.time()
    resp2 = llm_client.chat(messages, temperature=0.7, max_tokens=200)
    latencies.append(time.time() - start)
    stories.append(resp2)
    print(f"\nTurn 2 ({latencies[1]:.2f}s): {resp2[:150]}...")
    
    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies)
    distinct_2 = distinct_n(stories, 2)
    
    print(f"\n📊 Consistency Metrics:")
    print(f"   - Average latency: {avg_latency:.2f}s")
    print(f"   - Distinct-2 (diversity): {distinct_2:.4f}")
    
    return avg_latency, distinct_2

def main():
    print("\n" + "=" * 60)
    print("🚀 LOCAL MODEL DEPLOYMENT EVALUATION")
    print("=" * 60)
    print(f"Model: {llm_client._settings.OPENAI_MODEL}")
    print(f"Base URL: {llm_client._settings.OPENAI_BASE_URL}")
    
    results = {}
    
    try:
        # Test 1: Story generation
        story, story_latency, story_metrics = test_story_generation()
        results["story_latency"] = story_latency
        results.update(story_metrics)
        
        # Test 2: Option generation
        json_success, json_latency = test_option_generation()
        results["json_success"] = json_success
        results["json_latency"] = json_latency
        
        # Test 3: Consistency
        avg_lat, distinct_2 = test_consistency()
        results["avg_turn_latency"] = avg_lat
        results["multi_turn_distinct_2"] = distinct_2
        
        # Final summary
        print("\n" + "=" * 60)
        print("📋 EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\n⏱️  Performance:")
        print(f"   - Story generation latency: {results['story_latency']:.2f}s")
        print(f"   - JSON generation latency: {results['json_latency']:.2f}s")
        print(f"   - Average turn latency: {results['avg_turn_latency']:.2f}s")
        
        print(f"\n📊 Quality Metrics:")
        print(f"   - Distinct-1: {results['distinct_1']:.4f}")
        print(f"   - Distinct-2: {results['distinct_2']:.4f}")
        print(f"   - Type-Token Ratio: {results['type_token_ratio']:.4f}")
        print(f"   - Flesch Reading Ease: {results['flesch_reading_ease']:.2f}")
        
        print(f"\n✅ Functionality:")
        print(f"   - JSON parsing: {'PASS' if results['json_success'] else 'FAIL'}")
        
        # Save report
        report_path = Path("reports/simple_eval_report.md")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write("# Simple Model Evaluation Report\n\n")
            f.write(f"**Model:** {llm_client._settings.OPENAI_MODEL}\n")
            f.write(f"**Base URL:** {llm_client._settings.OPENAI_BASE_URL}\n\n")
            
            f.write("## ⏱️ Performance\n")
            f.write(f"- **Story generation latency:** {results['story_latency']:.2f}s\n")
            f.write(f"- **JSON generation latency:** {results['json_latency']:.2f}s\n")
            f.write(f"- **Average turn latency:** {results['avg_turn_latency']:.2f}s\n\n")
            
            f.write("## 📊 Quality Metrics\n")
            f.write(f"- **Distinct-1:** {results['distinct_1']:.4f}\n")
            f.write(f"- **Distinct-2:** {results['distinct_2']:.4f}\n")
            f.write(f"- **Type-Token Ratio:** {results['type_token_ratio']:.4f}\n")
            f.write(f"- **Flesch Reading Ease:** {results['flesch_reading_ease']:.2f}\n\n")
            
            f.write("## ✅ Functionality\n")
            f.write(f"- **JSON parsing:** {'PASS' if results['json_success'] else 'FAIL'}\n\n")
            
            f.write("## 📖 Sample Generated Story\n")
            f.write(f"```\n{story[:1000]}\n```\n")
        
        print(f"\n💾 Report saved to: {report_path}")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
