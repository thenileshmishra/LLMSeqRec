<!-- source llm_env/bin/activate -->

# Train both models (this will generate logs)
python -m LLMSeqRec.train.train_llmseqrec
python -m LLMSeqRec.train.train_sasrec

# Generate plots
python -m LLMSeqRec.eval.analysis
