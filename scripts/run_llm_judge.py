#!/usr/bin/env python3
"""Run LLM judge evaluation on a sample story using current model configuration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.llm_judge import judge

sample_transcript = """Narrator: You awaken in the root-choked hollow of an ancient oak, its bark gnarled like the knuckles of a sleeping giant. Moonlight filters through a canopy of silver leaves, casting shifting patterns on the mossy ground where you lie. The air is cool and carries the scent of damp earth and distant rain, while the forest around you hums with the chorus of crickets and the occasional hoot of an unseen owl. A faint, ethereal glow pulses from a crack in the tree's trunk, revealing a sliver of light within.

Pushing yourself up, you notice a small, leather-bound journal tucked into a hollow near your feet, its pages filled with sketches of strange symbols and the phrase "Follow the whispering stones" repeated in hurried script. The journal feels warm to the touch, as if it holds some residual magic. From deeper in the woods, you hear a soft, melodic whispering that seems to call your name, weaving between the trees like a thread of silver mist.

The choice lies before you: do you open the journal to study its cryptic notes, or do you heed the whispering and step into the shadowed path that winds deeper into the forest?

Player: Reach out and touch the faint, ethereal glow pulsing from the crack in the tree's trunk.

Narrator: You reach out, your fingers brushing the crack in the ancient oak's trunk. The moment your skin makes contact, a slick, viscous light oozes like warmed honey over your hand. It smells of split open pine sap and something cloyingly sweet, like overripe fruit left to rot in the sun. The glow clings to your skin with an unpleasant warmth, pulsing like a slow, sickly heartbeat against your palm. A faint shiver of revulsion runs through you as the light seems to writhe and coil around your fingers, refusing to let go.

The tree's bark groans under your touch, a deep, guttural sound that vibrates through the ground and up your arm. The silver leaves above rustle violently, though no wind blows, and the ethereal glow from the crack intensifies, casting long, distorted shadows that seem to reach for you. The melodic whispering from the forest deepens, its tone shifting from a call to a low, mocking chuckle that seems to echo from the very wood of the oak itself. The journal in your other hand grows warmer, its leather cover softening as if it, too, is reacting to the tree's strange secretion.

Player: Follow the darkening path deeper into the forest, where the shadows seem to thicken and wait.

Narrator: You step onto the darkening path, your feet sinking slightly into the softened, slick moss. The air grows colder, the scent of damp earth and decaying roots thickening into a palpable miasma that clings to the back of your throat. The shadows between the trees deepen, stretching like grasping fingers across the trail, and the whispering from the ancient oak follows you, its mocking tone now a constant, low drone that seems to vibrate in your bones. The greasy residue on your hand glows faintly, a sickly beacon in the gloom, and the journal in your other hand pulses with a warmth that feels more like a fever than a comfort.

Each step is a deliberate, fearful choice, the path narrowing as twisted roots claw at your boots and thorny branches snag at your clothes. The whispering words echo louder now, "The stones remember the blood," looping in your mind like a curse. You catch glimpses of smooth, grey stones half-buried in the undergrowth ahead, their surfaces reflecting the faint, unnatural light from your hand. The forest seems to close in around you, the canopy above knitting into a roof of darkness that blots out the moon, and the only sound is your own ragged breathing and the relentless, sibilant whisper that promises secrets drenched in violence.
"""

print("=" * 60)
print("Running LLM Judge Evaluation")
print("=" * 60)
print(f"Model: {sys.argv[1] if len(sys.argv) > 1 else 'current config'}")
print()

scores = judge(sample_transcript)

print("\n📊 LLM Judge Scores:")
print("-" * 40)
for key, value in scores.items():
    print(f"  {key}: {value}")

print("\n✅ Evaluation complete!")
