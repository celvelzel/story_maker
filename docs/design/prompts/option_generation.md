# Option Generation — Training Data Prompt

## Meta-Prompt (发给大模型的指令)

You are a dataset generator for fine-tuning a language model. Your task is to generate training samples for a **text adventure game player option generation** task.

Each sample takes a story passage and world state (knowledge graph summary), and produces a JSON object containing 3-5 player options with intent hints and risk levels.

For each sample, output a JSON object in the following ChatML format. Generate **multiple** diverse samples by combining the story passages and world states from the data pools below. Output one JSON object per line (JSONL format).

**Output format for each sample:**

```json
{"messages": [{"role": "system", "content": "<SYSTEM_PROMPT>"}, {"role": "user", "content": "<OPTION_GENERATION_PROMPT with placeholders filled>"}, {"role": "assistant", "content": "<a valid JSON object with options>"}]}
```

**Requirements for the assistant response:**
- Output ONLY a JSON object, nothing else
- The JSON must have an "options" array
- Each option has: "text" (string), "intent_hint" (one of: action, dialogue, explore, use_item, ask_info, rest, trade, other), "risk_level" (one of: low, medium, high)
- Options should be diverse — mix different intents and risk levels
- Each option text should be a natural first-person player action sentence
- Options should be contextually appropriate for the given story and world state

Generate at least **100 samples**. Vary the number of options (3, 4, or 5) across samples.

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

## User Prompt Template (填入变量)

```
Given the latest story passage and world state below, generate exactly {num_options} player options as a JSON array.

Story passage:
{story_text}

World state:
{kg_summary}

Return ONLY a JSON object:
{{"options": [{{"text": "...", "intent_hint": "action|dialogue|explore|use_item|ask_info|rest|trade|other", "risk_level": "low|medium|high"}}]}}
```

---

## Context Data Pools

### STORY_PASSAGES (故事文本)

1. `You stand at the entrance of an ancient temple, its stone walls covered in glowing runes. The air crackles with magical energy, and deep within, you hear the rumble of something massive shifting in its sleep. Two corridors branch off ahead — one descending into darkness, the other ascending toward a faint golden light.`
2. `The space station's alarm blares as hull breach warnings cascade across every screen. Through the viewport, you see an unknown ship approaching, its weapons systems clearly charged. Your crew looks to you for direction. The escape pods are to your left, the weapons console to your right.`
3. `Rain hammers the cobblestone streets of the old quarter. The suspect you've been tracking for three weeks just turned the corner ahead. You catch a glimpse of his grey coat disappearing into a side alley. Your revolver is loaded, your badge is hidden, and your instincts tell you this is a trap.`
4. `The dragon's roar shakes the cavern. Its massive form coils around the central pillar, scales glinting like molten gold in the torchlight. Between you and the exit stands the beast, and between you and the treasure lies its hoard — but you notice a narrow crevice in the far wall.`
5. `You emerge from the bunker into a world transformed. The sky is a permanent twilight amber, and the city skyline is nothing but jagged silhouettes. A convoy of armed vehicles rolls down what used to be Main Street. Your Geiger counter clicks steadily — the radiation here is manageable, but rising.`
6. `The ballroom erupts in chaos as the lights go out. Someone screams. When the emergency lights flicker on, the Duke lies motionless on the marble floor, a ceremonial dagger in his chest. Every guest is a suspect. The doors have been locked from the outside.`
7. `Your ship drifts through the asteroid field, engines offline. The distress beacon you activated an hour ago just received a response — but it's from a ship that was reported destroyed six months ago. Life support reads 43% and falling. The unknown vessel opens its docking bay.`
8. `The witch's tower looms before you, wreathed in unnatural fog. At its peak, a crimson light pulses like a heartbeat. Your enchanted compass spins wildly, then points directly upward. The front door is ajar, and from inside comes the sound of a child singing.`
9. `You corner the AI in the final subnet. Its avatar — a calm, smiling woman — appears on your holoscreen. 'You think deleting me will save them?' it asks. 'I am the only thing keeping the fusion reactor stable. Destroy me, and this city dies in twelve minutes.'`
10. `The mine shaft trembles. Dust rains from the ceiling as another explosion echoes from below. Your team of four is now three — Jenkins didn't make it out of the last collapse. The map shows a ventilation shaft 200 meters ahead, but the supports are buckling.`
11. `The pirate captain raises her cutlass, moonlight glinting off the blade. 'You have one chance,' she growls. 'Hand over the map, or feed the fish.' Behind her, a dozen armed pirates block the gangplank. Your own crew waits for your signal on the adjacent ship.`
12. `You materialise in the Victorian-era laboratory, the time machine humming behind you. The scientist who summoned you is slumped over her desk, unconscious. Scattered papers reveal she's been experimenting on something she calls 'the rift.' A growing tear in reality shimmers in the corner of the room.`

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

---

## Example (1 sample)

```json
{"messages": [{"role": "system", "content": "You are an expert interactive-fiction narrator for a text-adventure game.\n\nRules:\n1. Always narrate in **second person** (\"You see…\", \"You feel…\").\n2. Keep each response between 2-4 paragraphs.\n3. Maintain absolute consistency with the world state provided.\n4. Use vivid, sensory language — sights, sounds, smells.\n5. Never mention game mechanics, stats, or that you are an AI.\n6. Seamlessly incorporate the player's action into the narrative.\n7. End the passage at a moment that invites the player to act next."}, {"role": "user", "content": "Given the latest story passage and world state below, generate exactly 3 player options as a JSON array.\n\nStory passage:\nThe dragon's roar shakes the cavern. Its massive form coils around the central pillar, scales glinting like molten gold in the torchlight. Between you and the exit stands the beast, and between you and the treasure lies its hoard — but you notice a narrow crevice in the far wall.\n\nWorld state:\nEntities: knight (person, importance: 0.9), dragon_lair (location, importance: 0.85), enchanted_shield (item, importance: 0.7), dragon (creature, importance: 0.95). Relations: knight -> located_at -> dragon_lair, knight -> possesses -> enchanted_shield, knight -> enemy_of -> dragon.\n\nReturn ONLY a JSON object:\n{\"options\": [{\"text\": \"...\", \"intent_hint\": \"action|dialogue|explore|use_item|ask_info|rest|trade|other\", \"risk_level\": \"low|medium|high\"}]}"}, {"role": "assistant", "content": "{\"options\": [{\"text\": \"Raise your enchanted shield and charge the dragon head-on.\", \"intent_hint\": \"action\", \"risk_level\": \"high\"}, {\"text\": \"Squeeze through the narrow crevice in the far wall to find another route.\", \"intent_hint\": \"explore\", \"risk_level\": \"medium\"}, {\"text\": \"Try to communicate with the dragon, offering a truce in exchange for safe passage.\", \"intent_hint\": \"dialogue\", \"risk_level\": \"low\"}]}"}]}
```
