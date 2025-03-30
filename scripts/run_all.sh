#!/bin/bash
# run_all.sh - Script to run training, validation, and analysis

# Print the start time
echo "Starting all experiments at: $(date)"

# 1. Train LLMSeqRec model
echo "Training LLMSeqRec model..."
python -m LLMSeqRec.train.train_llmseqrec
if [ $? -ne 0 ]; then
    echo "LLMSeqRec training failed. Exiting."
    exit 1
fi

# 2. Train SASRec baseline model
echo "Training SASRec baseline model..."
python -m LLMSeqRec.train.train_sasrec
if [ $? -ne 0 ]; then
    echo "SASRec training failed. Exiting."
    exit 1
fi

# 3. Validate LLMSeqRec model
echo "Validating LLMSeqRec model..."
python -m LLMSeqRec.eval.validate_llmseqrec
if [ $? -ne 0 ]; then
    echo "LLMSeqRec validation failed. Exiting."
    exit 1
fi

# 4. Validate SASRec baseline model
echo "Validating SASRec baseline model..."
python -m LLMSeqRec.eval.validate_sasrec
if [ $? -ne 0 ]; then
    echo "SASRec validation failed. Exiting."
    exit 1
fi

# 5. Generate analysis plots
echo "Generating analysis plots..."
python -m LLMSeqRec.eval.analysis
if [ $? -ne 0 ]; then
    echo "Analysis plotting failed. Exiting."
    exit 1
fi

echo "All experiments completed at: $(date)"
