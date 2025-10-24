# “ChatGPT Suggested This Fix and It Seems to Work”: What Code Comments Reveal about Human-AI Collaboration
Analysis of LLM Usage in Open Source Software Projects from Self Admitted Comments and Linked Commits


.
├─ annotations_500_task_contr/              # Human annotations (500 comments)
│  ├─ ...                                    # Includes A/B labels + resolved final labels
│
├─ DS-EM/                                    # Dawid–Skene EM over LLM annotators
│  ├─ Contribution Type/                      # Raw LLM votes for contribution labels
│  ├─ Task Type/                              # Raw LLM votes for task labels
│  └─ dawid_skene.py                          # DS-EM with human gold for calibration/eval
│
├─ data/
│  └─ combined_whole_dataset_per_comment.json.zip
│      # 12,043 comments + 6,912 intro commits + 2,948 first-change commits
│
└─ README.md

  
