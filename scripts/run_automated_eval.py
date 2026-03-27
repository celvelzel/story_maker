import os
import sys
import time
import json
import random
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.game_engine import GameEngine
from src.evaluation.metrics import full_evaluation
from src.evaluation.llm_judge import judge

def main():
    parser = argparse.ArgumentParser(description="Automated multi-turn evaluation for StoryWeaver")
    parser.add_argument("--turns", type=int, default=3, help="Number of turns to play")
    parser.add_argument("--genre", type=str, default="fantasy", help="Game genre")
    parser.add_argument("--output", type=str, default="reports/automated_eval_report.md", help="Output report file")
    
    args = parser.parse_args()
    
    print(f"Initializing GameEngine (Genre: {args.genre})...")
    engine = GameEngine(genre=args.genre)
    
    print("Starting game...")
    start_time = time.time()
    try:
        turn_result = engine.start_game()
    except Exception as e:
        print(f"Error starting game: {e}")
        return
        
    latency = time.time() - start_time
    
    texts = [turn_result.story_text]
    latencies = [latency]
    kg_turn_stats = [{"turn_id": 0, "node_count": engine.kg.graph.number_of_nodes(), "edge_count": engine.kg.graph.number_of_edges()}]
    turn_conflict_counts = [len(turn_result.nlu_debug.get("conflicts", []))]
    
    print(f"Turn 0 [Start] | Latency: {latency:.2f}s")
    
    transcript = f"Narrator: {turn_result.story_text}\n\n"
    
    for i in range(1, args.turns + 1):
        if turn_result.options:
            action = random.choice(turn_result.options).text
        else:
            action = "look around"
            
        print(f"Turn {i} | Player Action: {action}")
        transcript += f"Player: {action}\n\n"
        
        start_time = time.time()
        try:
            turn_result = engine.process_turn(action)
        except Exception as e:
            print(f"Error processing turn {i}: {e}")
            break
            
        latency = time.time() - start_time
        
        texts.append(turn_result.story_text)
        latencies.append(latency)
        kg_turn_stats.append({
            "turn_id": i, 
            "node_count": engine.kg.graph.number_of_nodes(), 
            "edge_count": engine.kg.graph.number_of_edges()
        })
        turn_conflict_counts.append(len(turn_result.nlu_debug.get("conflicts", [])))
        
        transcript += f"Narrator: {turn_result.story_text}\n\n"
        print(f"Turn {i} | Latency: {latency:.2f}s | Nodes: {engine.kg.graph.number_of_nodes()}")

    print("Running Full Evaluation Metrics...")
    entity_names = [node for node in engine.kg.graph.nodes()]
    
    try:
        metrics_result = full_evaluation(
            texts=texts,
            entity_names=entity_names,
            turn_conflict_counts=turn_conflict_counts,
            kg_turn_stats=kg_turn_stats
        )
    except Exception as e:
        print(f"Error running metrics: {e}")
        metrics_result = {}
    
    print("Running LLM Judge Evaluation...")
    try:
        judge_result = judge(transcript)
    except Exception as e:
        print(f"Error running LLM judge: {e}")
        judge_result = {}
    
    # Calculate Latency Stats
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    
    # Format the Report
    report = f"# Automated Evaluation Report\n\n"
    report += f"**Genre:** {args.genre} | **Turns:** {args.turns}\n\n"
    
    report += f"## ⏱️ Response Speed (Latency)\n"
    report += f"- **Average Latency:** {avg_latency:.2f}s\n"
    report += f"- **Max Latency:** {max_latency:.2f}s\n"
    report += f"- **Min Latency:** {min_latency:.2f}s\n\n"
    
    report += f"## 📊 Traditional Metrics\n"
    for k, v in metrics_result.items():
        report += f"- **{k}:** {v:.4f}\n"
    
    report += f"\n## ⚖️ LLM Judge Scores\n"
    for k, v in judge_result.items():
        report += f"- **{k}:** {v}\n"
        
    report += f"\n## 📜 Transcript\n"
    report += f"```text\n{transcript}\n```\n"
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\nReport generated successfully: {args.output}")

if __name__ == '__main__':
    main()
