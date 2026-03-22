# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit learn

You may complete this model card for whichever version you used, or compare both if you explored them.

## 1. Model Overview

**Model type:**  
I implemented and evaluated the **rule-based model** in `mood_analyzer.py`. The ML model architecture from `ml_experiments.py` uses scikit-learn's CountVectorizer with LogisticRegression, but for this card I focus on the rule-based version that is fully operational.

**Intended purpose:**  
This model is trying to classify short text messages as moods like positive, negative, neutral, or mixed.

**How it works (brief):**  
The rule-based model preprocesses text by normalizing punctuation, replacing emojis/emoticons with semantic tokens (smile_emoji, sad_emoji, etc.), and reducing repeated characters. It then scores sentiment by counting positive and negative words in a dictionary, applies negation handling ("not happy" flips sentiment), and assigns emojis fixed weights. The final label (positive/negative/neutral/mixed) is derived from the numeric score with special handling for balanced texts.



## 2. Data

**Dataset description:**  
The dataset contains **17 labeled posts** in `SAMPLE_POSTS` and `TRUE_LABELS`. The original starter set had 6 posts; I added 11 more examples to increase diversity. Posts range from 4 to 17 words. Labels include "positive", "negative", "neutral", and "mixed" (introduced to capture ambiguous emotion combinations).

**Labeling process:**  
Labels were assigned by analyzing the emotional content: positive sentiment for posts with optimistic language ("love", "great", "excited"), negative for distress or pessimism ("terrible", "stressed", "sad"), neutral for factual or emotionally flat statements, and mixed for posts that simultaneously express contradictory emotions. Challenging examples include "Feeling tired but kind of hopeful" (mixed: fatigue + hope) and "I absolutely love getting stuck in traffic 🙃" (sarcasm masking frustration as negative despite positive word choice). A few posts like "Honestly, not bad… could be worse I guess" genuinely sit between neutral and mild positive.

**Important characteristics of your dataset:**  

- **Contains slang and modern language**: "lowkey", "highkey", "no cap", "drained my soul"
- **Includes diverse emojis**: 🌎, 😂, 🥲, 💀, 🙃 mapped to semantic sentiment tokens
- **Contains emoticons**: :), :( coded as emoticon tokens for consistent handling
- **Includes sarcasm**: "I absolutely love getting stuck in traffic 🙃" (negative sentiment despite positive word)
- **Some posts express mixed feelings**: "Feeling weird… like happy-sad at the same time", "Feeling tired but kind of hopeful"
- **Contains short and long examples**: ranging from 4 to 17 words
- **Includes negation patterns**: "I am not happy", "not bad", "never fun"
- **Features subtle or ambiguous messages**: "This is fine" (neutral despite familiarity with anxious meme), "Honestly, not bad…" (lukewarm positivity)

**Possible issues with the dataset:**  

- **Small dataset**: Only 17 examples may not capture language diversity sufficiently
- **Label imbalance**: Positive (6), Negative (5), Mixed (3), Neutral (3) — positive overrepresented
- **Sarcasm ambiguity**: "I absolutely love getting stuck in traffic" is sarcasm but labeled negative; the rule-based model cannot reliably detect this
- **Cultural/linguistic bias**: Slang and emojis are modern internet conventions; may not generalize to formal speech or non-English contexts
- **Subjective mixed labels**: The distinction between "neutral" and "mixed" is fuzzy; different annotators might disagree
- **No explicit domain**: Posts are casual social media style; may not transfer to customer service, medical, or formal contexts

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  

- **Base sentiment scoring**: Positive words add +1 per occurrence; negative words subtract -1 per occurrence
- **Negation handling**: If a sentiment word is preceded by "not", "no", "never", "don't", "can't", "won't", "isn't", "wasn't", its sentiment flips (e.g., "not happy" becomes -1 instead of +1)
- **Emoji weights**: smile_emoji and laugh_emoji each add +1; sad_emoji, dead_emoji, sarcasm_emoji each subtract -1
- **No explicit word weighting**: All positive/negative words have equal weight; "hate" and "annoyed" both score -1
- **Label thresholds**:
  - Score > 0 → "positive"
  - Score < 0 → "negative"
  - Score == 0 → Check for presence of both positive and negative words:
    - If both present → "mixed"
    - Otherwise → "neutral"
- **Preprocessing normalization**: Repeated characters reduced ("soooo" → "soo"), punctuation stripped, emojis/emoticons tokenized

**Strengths of this approach:**  

- **Interpretability**: Every prediction is explainable via the `explain()` method showing which words and emojis contributed
- **Negation intuition**: Correctly handles "not happy" and similar reversals, which improves over naive word-counting
- **Emoji recognition**: Identifies and weights emojis as sentiment signals, useful for modern text
- **Predictable behavior**: No randomness; identical input always produces identical output
- **Fast**: Rule-based scoring is O(n) in token length; no training required
- **Reasonable emoji handling**: Correctly identifies laugh_emoji (😂) as positive, dead_emoji (💀) as negative
- **Good on simple, clear examples**: Posts like "I love this class so much" and "Today was a terrible day" are classified correctly

**Weaknesses of this approach:**  

- **Cannot detect sarcasm**: "I absolutely love getting stuck in traffic 🙃" contains "love" (positive) and even a sarcasm emoji, but the rule-based model scores it as neutral (emoji weight cancels word weight), missing the true negative sentiment
- **Limited context window**: Only checks previous token for negation; longer negation patterns like "On a scale of 1 to 10 how much do I NOT like this" are missed
- **Unknown slang**: Slang words like "lowkey", "highkey", "no cap", "drained" are not in the sentiment dictionary, so they contribute no score
- **Mixed emotion failures**: Posts with balanced positive/negative content sometimes default to neutral when score = 0, missing subtle leaning (e.g., "Feeling tired but kind of hopeful" has both words but scores as -1 from negation, labeled negative instead of mixed)
- **No contextual weighting**: "I hate this" and "I mildly dislike this" both score -1; no nuance
- **Emoji ambiguity**: The sarcasm emoji (🙃) is identified but may appear in non-sarcastic contexts and muddle the signal
- **Small training data impact**: The model rule set is fixed and brittle; data-driven ML might adapt better with more examples

## 4. How the ML Model Works (if used)

**Features used:**  
Bag of words using scikit-learn's `CountVectorizer`, which converts each post into a vector of word occurrence counts. No special preprocessing beyond sklearn defaults; this naive representation does not handle negation, emoticons, or emoji as the rule-based model does.

**Training data:**  
Both models train on the full `SAMPLE_POSTS` and `TRUE_LABELS` (17 examples). No held-out test set is used; accuracy is measured on training data only, which risks overfitting.

**Training behavior:**  
Not formally tested, but the rule-based model's accuracy (65% on all 17 posts) is stable and does not depend on the number of examples—the rules are fixed. The ML model with CountVectorizer + LogisticRegression would likely show improvement with more training data due to better feature coverage, but sklearn has not been benchmarked here.

**Strengths and weaknesses:**  
**Strengths**: ML models can automatically learn non-obvious patterns from labeled data; negation and emoji sentiment can be inferred without explicit rules. **Weaknesses**: With only 17 training examples, the CountVectorizer + LogisticRegression model will almost certainly overfit, learning spurious associations (e.g., "tired" only from one "mixed" post and one "negative" post, leading to brittle predictions on new data). No regularization or cross-validation is applied, so the true generalization error is unknown.

## 5. Evaluation

**How you evaluated the model:**  
Both versions evaluated on the full labeled dataset (SAMPLE_POSTS + TRUE_LABELS) in a simple accuracy measurement: (# correct predictions) / (total examples).

**Rule-Based Model Accuracy**: **0.65 (65%)** on all 17 posts—11 correct, 6 incorrect.

**Examples of correct predictions:**  

1. **"I love this class so much"** → predicted: positive, true: positive  
   Explanation: Strong positive sentiment from "love" (+1) and "class" association; simple, unambiguous.

2. **"Today was a terrible day"** → predicted: negative, true: negative  
   Explanation: "terrible" is a negative keyword (-1); straightforward negative emotion.

3. **"I am not happy about this"** → predicted: negative, true: negative  
   Explanation: Negation detection: "not" precedes "happy", so the +1 from "happy" is flipped to -1; final score = -1 → negative. Negation rule works as intended.

**Examples of incorrect predictions:**  

1. **"Feeling tired but kind of hopeful"** → predicted: negative, true: mixed  
   Explanation: "tired" is a negative word (-1) and preempts the positive sentiment from "hopeful" (+1). The negation rule misinterprets the sentence structure. Expected: mixed because both sentiments are reasonably balanced; actual: score = 0 but only "tired" is detected, not recognized as mixed. Root cause: limited context window and no compound emotion parsing.

2. **"I absolutely love getting stuck in traffic 🙃"** → predicted: neutral, true: negative  
   Explanation: Sarcasm. The model sees "love" (+1) and sarcasm_emoji (-1), which cancel to 0, then recognizes no pure negation pattern. The irony of loving traffic is invisible to rule-based scoring. Correct label is negative (frustrated). The model cannot detect sarcasm.

3. **"Why does everything have to be so complicated"** → predicted: neutral, true: negative  
   Explanation: The word "complicated" is not in the sentiment dictionary, so it contributes 0 to the score. The rhetorical frustration is missed. If "complicated" were labeled negative, or if the question structure were recognized as expressing distress, this would score correctly.

## 6. Limitations

- **Very small dataset (17 posts)**: Not sufficient to train robust ML models; rule-based model is brittle and hand-tuned for this specific set
- **No sarcasm detection**: Critical failure mode for mood detection; sarcasm inverts sentiment but rule-based model is blind to it
- **Limited word dictionary**: Only 10 positive and 10 negative words; many sentiment-bearing words ("complicated", "amazing", "drained") are missing
- **No negation handling beyond immediate predecessor**: Multi-token negations like "not really happy" or "don't think I like" are imperfectly captured
- **No contrastive learning**: Posts that say "I'm sad but I'm trying to be positive" genuinely have mixed emotion, but rule-based scoring may collapse to one label
- **Emoji coverage**: Only 5 emoji types are recognized; others pass through as punctuation noise
- **No test set**: Accuracy is measured on training data only; true generalization performance is unknown and likely lower
- **Sensitivity to preprocessing**: Character normalization ("soooo" → "soo") may lose expressions of intensity
- **Hardcoded thresholds**: Label boundaries (score > 0 vs. score < 0) are fixed; no tuning or adaptive behavior
- **English-only**: The dictionary and rules are English-specific; no multilingual support

## 7. Ethical Considerations

- **Misclassifying distress as neutral**: Posts expressing serious emotional distress (e.g., "Why does everything have to be so complicated") may be classified as neutral, potentially causing harm if used in mental health monitoring or crisis detection systems
- **Sarcasm and irony inversion**: "I absolutely love getting stuck in traffic" is misclassified as neutral instead of negative frustration, risking inappropriate responses in customer service or sentiment-driven decision systems
- **Language bias**: The word lists and rules reflect modern English internet culture (slang, emojis) and do not represent formal English, non-English languages, or marginalized communities. A person speaking differently may have their emotions consistently misunderstood
- **Overfitting to training data**: With only 17 examples, the model may reflect the biases and word choices of whoever labeled those 17 posts, not general human emotion
- **Privacy concerns**: If deployed to analyze personal messages (e.g., email, chat, social media), mood detection raises privacy and consent issues. Users may not expect their emotional tone to be automatically classified
- **Feedback loops**: If a system uses mood predictions to rankk or filter content (e.g., "don't show this negative post"), errors could amplify: users never see their actual mood reflected back, causing further misunderstanding
- **Lack of transparency**: Rule-based predictions are explainable (via `explain()`), but if deployed to end users without showing the reasoning, they may appear arbitrary and erode trust

## 8. Ideas for Improvement

**For the rule-based model:**
- **Expand sentiment dictionary**: Add "complicated", "drained", "stressed", and other words discovered as failures; weight important words higher (e.g., "hate" > "annoyed")
- **Improve negation handling**: Look further back and handle compound negations ("not really happy", "don't think I like")
- **Add sarcasm detection**: Train a secondary classifier or use cues like contradiction ("love" + emoji 🙃) as sarcasm signals
- **Contextual emoji weighting**: Increase weight of 💀 (dead_emoji) from -1 to -2 if it appears with other negative words
- **Tune label thresholds**: Use threshold search to optimize accuracy (e.g., score >= 2 for "positive" instead of > 0)

**For the ML model:**
- **Expand training data**: Collect 100–500 labeled posts to reduce overfitting and improve generalization
- **Use TF-IDF**: Replace CountVectorizer with TfidfVectorizer to down-weight common words and highlight rare sentiment signals
- **Add feature engineering**: Prepend the rule-based preprocessing (emoji tokenization, negation markers) so the ML model has richer input
- **Cross-validation**: Implement k-fold cross-validation to measure true generalization performance
- **Regularization**: Add L1/L2 regularization to LogisticRegression to reduce overfitting
- **Upgrade model**: Try a small neural network (1–2 hidden layers) or a pre-trained embeddings (GloVe, Word2Vec) to capture semantic similarity

**For both:**
- **Separate test set**: Hold out 20–30% of data for unbiased evaluation; train on remaining 70–80%
- **Sarcasm dataset**: Curate a labeled sarcasm subset and evaluate separately
- **Human evaluation**: Have multiple annotators label examples and report inter-annotator agreement
- **Error analysis**: Categorize failures (sarcasm, negation, slang, etc.) and prioritize by frequency
- **Domain-specific tuning**: If deploying to customer service, retrain/retune on customer message data
- **Multilingual expansion**: Add support for Spanish, French, or other languages with translated word lists and emoji mappings
- **Explainability UI**: If deployed, show users the `explain()` output so they can understand and correct misclassifications
