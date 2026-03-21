# Story Continuation Generation — Training Data Prompt

## Meta-Prompt (发给大模型的指令)

You are a dataset generator for fine-tuning a language model. Your task is to generate training samples for a **text adventure game story continuation** task.

Each sample takes a world state (knowledge graph summary), recent history, player intent, player input, and emotional tone, and produces a 2-4 paragraph story continuation.

For each sample, output a JSON object in the following ChatML format. Generate **multiple** diverse samples by randomly combining context variables from the data pools below. Output one JSON object per line (JSONL format).

**Output format for each sample:**

```json
{"messages": [{"role": "system", "content": "<SYSTEM_PROMPT>"}, {"role": "user", "content": "<STORY_CONTINUE_PROMPT with all placeholders filled>"}, {"role": "assistant", "content": "<a vivid 2-4 paragraph story continuation>"}]}
```

**Requirements for the assistant response:**
- Write in **second person** ("You see…", "You feel…")
- 2-4 paragraphs
- React to the player's action and emotional tone
- Adjust pacing, word choice, and atmosphere to match the emotion
- Advance the plot
- Maintain consistency with the provided world state and history
- End at a moment that invites the player to act next

Generate at least **100 samples**. Mix and match the context variables freely to create diverse scenarios.

---

## System Prompt (固定，复制到每个 sample 的 system role)

```
You are an expert interactive-fiction narrator for a text-adventure game.

Rules:
1. Always narrate in **second person** ("You see…", "You feel…").
2. Keep each response between 2-4 paragraphs.
3. Maintain absolute consistency with the world state provided.
4. Use vivid, sensory language — sights, sounds, smells.
5. Never mention game mechanics, stats, or that you are an AI.
6. Seamlessly incorporate the player's action into the narrative.
7. End the passage at a moment that invites the player to act next.
```

---

## User Prompt Template (填入所有变量)

```
{kg_summary}

Recent history:
{history}

The player's intent is "{intent}".
The player's emotional tone is: {emotion}
The player says: "{player_input}"

Continue the story in 2-4 vivid paragraphs, reacting to the player's action and emotional tone. Adjust the narrative mood to match: a {emotion} tone should influence pacing, word choice, and atmosphere. Advance the plot.
```

---

## Context Data Pools

### KG_SUMMARIES (知识图谱摘要)

1. `Entities: hero (person, importance: 0.9), dark_forest (location, importance: 0.7), ancient_sword (item, importance: 0.8). Relations: hero -> located_at -> dark_forest, hero -> possesses -> ancient_sword.`
2. `Entities: captain (person, importance: 0.85), starship (item, importance: 0.8), nebula_station (location, importance: 0.6). Relations: captain -> possesses -> starship, starship -> located_at -> nebula_station.`
3. `Entities: detective (person, importance: 0.9), warehouse (location, importance: 0.5), mysterious_note (item, importance: 0.7). Relations: detective -> located_at -> warehouse, detective -> possesses -> mysterious_note.`
4. `Entities: merchant (person, importance: 0.6), harbor_town (location, importance: 0.8), treasure_map (item, importance: 0.9). Relations: merchant -> located_at -> harbor_town, treasure_map -> located_at -> harbor_town.`
5. `Entities: survivor (person, importance: 0.9), ruined_city (location, importance: 0.7), supplies_cache (item, importance: 0.8), raider_leader (creature, importance: 0.6). Relations: survivor -> located_at -> ruined_city, supplies_cache -> located_at -> ruined_city.`
6. `Entities: wizard (person, importance: 0.95), crystal_tower (location, importance: 0.8), spell_book (item, importance: 0.7), shadow_demon (creature, importance: 0.85). Relations: wizard -> located_at -> crystal_tower, wizard -> possesses -> spell_book, wizard -> enemy_of -> shadow_demon.`
7. `Entities: hacker (person, importance: 0.9), neon_alley (location, importance: 0.6), data_chip (item, importance: 0.85), mega_corp_security (person, importance: 0.7). Relations: hacker -> located_at -> neon_alley, hacker -> possesses -> data_chip, mega_corp_security -> enemy_of -> hacker.`
8. `Entities: explorer (person, importance: 0.8), ancient_ruins (location, importance: 0.9), golden_idol (item, importance: 0.95), trap_mechanism (item, importance: 0.5). Relations: explorer -> located_at -> ancient_ruins, golden_idol -> located_at -> ancient_ruins, trap_mechanism -> prevents -> explorer.`
9. `Entities: knight (person, importance: 0.9), dragon_lair (location, importance: 0.85), enchanted_shield (item, importance: 0.7), dragon (creature, importance: 0.95). Relations: knight -> located_at -> dragon_lair, knight -> possesses -> enchanted_shield, knight -> enemy_of -> dragon.`
10. `Entities: pilot (person, importance: 0.88), asteroid_belt (location, importance: 0.6), distress_signal (event, importance: 0.9), unknown_vessel (item, importance: 0.75). Relations: pilot -> located_at -> asteroid_belt, distress_signal -> caused_by -> unknown_vessel.`
11. `Entities: thief (person, importance: 0.85), grand_museum (location, importance: 0.8), ruby_pendant (item, importance: 0.9), guard_patrol (person, importance: 0.6). Relations: thief -> located_at -> grand_museum, ruby_pendant -> located_at -> grand_museum, guard_patrol -> prevents -> thief.`
12. `Entities: witch (person, importance: 0.9), haunted_manor (location, importance: 0.85), cursed_mirror (item, importance: 0.8), ghost_child (creature, importance: 0.7). Relations: witch -> located_at -> haunted_manor, witch -> possesses -> cursed_mirror, cursed_mirror -> related_to -> ghost_child.`

### HISTORIES (最近对话历史，可取 1-3 段拼接)

1. `[Player] I step into the dark forest.\n[Narrator] The canopy above swallows the daylight. Twisted roots snake across the path, and the air is thick with the scent of damp moss and decay. Ahead, a faint orange glow flickers between the trees.`
2. `[Player] I examine the control panel.\n[Narrator] The console hums to life, casting pale blue light across your face. Warning indicators flash in an unfamiliar script. A holographic map materialises, showing three adjacent corridors — one marked in red.`
3. `[Player] I ask the bartender about the missing person.\n[Narrator] The bartender's eyes dart toward the door before he leans in. 'She came through here three nights ago. Had a scar across her left cheek. Left with a man in a grey coat — said something about the old lighthouse.'`
4. `[Player] I charge at the goblin!\n[Narrator] Your blade arcs through the air, catching the creature off guard. It stumbles backward, screeching, and knocks over a stack of crates. Two more goblins emerge from the shadows, brandishing rusty daggers.`
5. `[Player] I activate the ancient mechanism.\n[Narrator] Gears grind and stone slabs shift. A spiral staircase descends into darkness below. Cold air rushes upward, carrying the distant echo of dripping water and something else — a low, rhythmic chanting.`
6. `[Player] I try to open the locked chest.\n[Narrator] The lock resists your first attempt. On the second try, something clicks. The lid creaks open to reveal a velvet pouch containing three silver coins and a folded letter sealed with black wax.`
7. `[Player] I search the victim's office.\n[Narrator] The desk drawers are a mess — papers scattered, a drawer forced open. You find a half-written resignation letter and a photograph of a man you don't recognise, standing in front of a warehouse on the waterfront.`
8. `[Player] I cast a protective ward.\n[Narrator] Runes shimmer into existence around you, forming a translucent barrier. The shadow tendrils recoil, hissing. Through the barrier you see the creature's form shift and reform, growing larger.`
9. `[Player] I hide behind the fallen pillar.\n[Narrator] You press yourself against the cold stone. Footsteps approach — heavy, deliberate. A beam of light sweeps across the floor inches from your hiding spot. Then silence. Then a voice: 'I know you're in here.'`
10. `[Player] I negotiate with the alien envoy.\n[Narrator] The envoy tilts its head, bioluminescent patterns rippling across its skin. It speaks in a series of clicks and low tones, which your translator renders as: 'Your offer is... acceptable. But we require proof of your claim.'`
11. `[Player] I rest by the campfire.\n[Narrator] The flames crackle and dance, casting long shadows against the canyon walls. You feel warmth seeping back into your bones. Above, unfamiliar constellations wheel slowly across the sky. The night is quiet — too quiet.`
12. `[Player] I follow the blood trail.\n[Narrator] The drops lead down a narrow alley, growing darker and more frequent. At the end, a door stands ajar. From inside, you hear laboured breathing and the metallic clink of something being dragged across the floor.`
13. `[Player] I attempt to hack the terminal.\n[Narrator] Lines of code scroll past as you bypass the outer firewall. The system is more sophisticated than expected. A progress bar inches forward — 47%... 63%... Then a warning pops up: 'Intrusion detected. Countermeasures initiated.'`
14. `[Player] I pick up the strange artifact.\n[Narrator] The moment your fingers close around it, a pulse of energy surges through your arm. The runes on its surface glow with an inner light. The air around you thickens, and for a heartbeat, you see — another place. Another time.`
15. `[Player] I confront the mayor about the cover-up.\n[Narrator] The mayor's composure cracks. A vein pulses at his temple. 'You don't understand what you're meddling with,' he says, his voice low and dangerous. 'Some doors, once opened, cannot be closed again.'`

### INTENTS (玩家意图)

- action
- dialogue
- explore
- use_item
- ask_info
- rest
- trade
- other

### EMOTIONS (玩家情绪)

- neutral
- excited
- anxious
- angry
- sad
- curious
- determined
- fearful
- joyful
- suspicious
- desperate
- hopeful

### PLAYER_INPUTS (玩家输入)

1. `I cautiously step forward into the darkness.`
2. `I draw my weapon and prepare for combat.`
3. `I try to communicate peacefully.`
4. `I search the area for useful items.`
5. `I retreat and find another path.`
6. `I use my special ability.`
7. `I call out to see if anyone responds.`
8. `I examine the nearest object closely.`
9. `I take cover and observe the situation.`
10. `I attempt to negotiate.`
11. `I press the mysterious button on the wall.`
12. `I follow the sound of running water.`
13. `I climb up to get a better vantage point.`
14. `I set a trap for whatever is ahead.`
15. `I use the artifact I found earlier.`
16. `I run as fast as I can toward the exit.`
17. `I try to reason with the creature.`
18. `I hide and wait for the right moment.`
19. `I check my equipment and prepare.`
20. `I shout a challenge into the darkness.`
21. `I tend to my wounded companion.`
22. `I activate the emergency protocol.`
23. `I pick the lock on the heavy door.`
24. `I follow the trail of clues.`
25. `I set up a defensive perimeter.`
26. `I distract the enemy while my ally flanks.`
27. `I surrender and see what they want.`
28. `I attempt to hack the security system.`
29. `I sacrifice my shield to break through.`
30. `I read the ancient inscription aloud.`

---

## Example (1 sample)

```json
{"messages": [{"role": "system", "content": "You are an expert interactive-fiction narrator for a text-adventure game.\n\nRules:\n1. Always narrate in **second person** (\"You see…\", \"You feel…\").\n2. Keep each response between 2-4 paragraphs.\n3. Maintain absolute consistency with the world state provided.\n4. Use vivid, sensory language — sights, sounds, smells.\n5. Never mention game mechanics, stats, or that you are an AI.\n6. Seamlessly incorporate the player's action into the narrative.\n7. End the passage at a moment that invites the player to act next."}, {"role": "user", "content": "Entities: knight (person, importance: 0.9), dragon_lair (location, importance: 0.85), enchanted_shield (item, importance: 0.7), dragon (creature, importance: 0.95). Relations: knight -> located_at -> dragon_lair, knight -> possesses -> enchanted_shield, knight -> enemy_of -> dragon.\n\nRecent history:\n[Player] I cast a protective ward.\n[Narrator] Runes shimmer into existence around you, forming a translucent barrier. The shadow tendrils recoil, hissing. Through the barrier you see the creature's form shift and reform, growing larger.\n\nThe player's intent is \"action\".\nThe player's emotional tone is: determined\nThe player says: \"I draw my weapon and prepare for combat.\"\n\nContinue the story in 2-4 vivid paragraphs, reacting to the player's action and emotional tone. Adjust the narrative mood to match: a determined tone should influence pacing, word choice, and atmosphere. Advance the plot."}, {"role": "assistant", "content": "Your hand closes around the hilt of your sword, the enchanted metal humming against your palm as if it senses the battle to come. The protective ward flickers — it won't hold much longer against the dragon's corrosive breath. You steady your stance, boots scraping against the ancient stone floor, and raise the enchanted shield. Its surface catches the dragon's firelight, casting dancing reflections across the cavern walls.\n\nThe dragon descends from its perch with a roar that shakes loose stalactites from the ceiling. Its eyes — twin pools of molten amber — lock onto you with predatory intelligence. A wave of heat washes over your face as it exhales, the air shimmering with barely contained flame. You feel your jaw tighten. Fear is a luxury you cannot afford.\n\nYou lunge forward, shield raised to deflect the first gout of fire. The enchanted barrier absorbs the blast with a high-pitched whine, but you feel the heat sear through to your arm beneath. The dragon's massive claw swipes down — you roll to the right, the talons gouging deep furrows in the stone where you stood a heartbeat ago. You come up swinging, your blade scoring a line of sparks across the beast's underbelly. It shrieks — a sound like tearing metal — and recoils, giving you a precious moment to reassess."}]}
```
