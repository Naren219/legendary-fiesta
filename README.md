## details

used Qwen2.5-0.5B-Instruct pretrained model.

This program synthesis task involves figuring out the sentence (aka program) that maps one list of numbers to another.
[[[5, 5, 8], [8]], [[3, 4], [4]], ...] --> input and output pairs

e.g. target program: filter_(and_(or_(gt(2),gt(2)),or_(even,even)))

This is similar in style to ARC-AGI puzzles, yet there's a lot more data available.

In the code, I did supervised fine-tuning via the HuggingFace transforms library and did Reinforcement Learning on top of that. 

Method: GRPO
Reward function(s):
- Only used one reward function (for correctness) as the training times otherwise were too long (might try integrating more in the future).

## Results
SFT Only
accuracy: 18%
SFT + RL
accuracy: 57%
SFT + RL (sampling 32 programs)
accuracy: 78%

Funny thing is that the reward was decreasing the whole time. It was quite noisy, but it's weird that the results proved it did better with RL.
<img width="846" height="453" alt="download" src="https://github.com/user-attachments/assets/5cf6af3f-3d49-4c60-963e-f73f9b87d526" />

Here's what Uncle Opus 4.5 told me:
```
The reward curve lied to you. Here's why:

**1. Training reward ≠ Eval accuracy**

During GRPO training:
- Model **samples** with temperature (exploration)
- Many samples fail → low reward
- But the **policy** (greedy decoding) improves underneath

At eval time:
- You use greedy decoding (or lower temp)
- Model picks its best guess
- Accuracy is higher

**2. Per-batch noise**

Your reward is sparse: program works (1.0) or doesn't (~0.0). With `num_generations=8` per batch, a few failures tank the batch average. The curve is just noise on top of actual learning.

**3. Harder examples later?**

If training data wasn't perfectly shuffled, later batches might have harder tasks → lower rewards even with a better model.

**4. What actually happened**

Step 0:      Model guesses randomly → some lucky hits → ~0.8 reward
Step 5000:   Model learned patterns → but explores bad paths → ~0.2 reward  
Eval:        Model uses learned patterns (greedy) → 57% accuracy

**The takeaway:**

For program synthesis, **ignore the reward curve**. Only trust held-out eval accuracy. Your GRPO run was a success — 3x improvement over SFT.
```

thank you https://github.com/thelogicalgrammar/arc-course/blob/main/labs/lab3.ipynb for most of the code!
