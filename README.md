
## 📂 Repository Structure

```
.
├──Annotation
│   ├──annotations_500_task_contr.csv        # Human annotations (500 comments)
|   ├──Instructions for Open-Coding Annotation and Developing a Taxonomy of Developer LLM Usage from GitHub Code Comments.pdf        # Annotation Instruction
├── DS-EM/                                # Dawid–Skene EM aggregation for LLM annotations
│   ├── Contribution Type/                # LLM votes for contribution labels
│   ├── Task Type/                        # LLM votes for task labels
│   └── dawid_skene.py                    # DS-EM script using human annotations as gold set for aggregating LLM annotations
│   ├──
├── Prompts                                # Contains prompt for LLM annotations
│   ├── Prompt_for_task_type_annotations.pdf
│   └── Prompt_for_contribution_type_annotatinos.pdf
├──Dataset
│   └── combined_whole_dataset_per_comment.json.zip
│       # 12,043 comments + 6,912 intro commits + 2,948 first-change commits
└── README.md
```

---

## 🧩 Contents Overview

### **Human Annotations** (`annotations_500_task_contr`)

Two annotators (**A** & **B**) labeled each comment for:

* **Task Type**
* **Contribution Type**

Includes final adjudicated labels after disagreement resolution.

**Task Type Labels**

* Code Implementation
* Code Enhancement
* Bug Identification & Fixing
* Testing
* Documentation
* Generic Mention & Indeterminate

**Contribution Type Labels**

* Implementation
* Knowledge and Concept Support
* Artifact Generation
* Generic Mention & Indeterminate

---

### **LLM Annotations + DS-EM** (`DS-EM/`)

* **Models:** `gpt-oss-20b` and `mistral-small-3.2`
* **Annotation Scopes:** Task Type and Contribution Type
* **`dawid_skene.py`: * ** Implements the Dawid–Skene Expectation–Maximization (DS-EM) algorithm for semi-supervised label aggregation, using human annotations as the gold set. It estimates true labels, annotator reliability (confusion matrices), and class priors to produce confidence-scored consensus annotations across multiple LLM annotators.

---

### **Prompts**

* Prompt_for_task_type_annotations.pdf  # For annotating task types
* Prompt_for_contribution_type_annotatinos.pdf # For annotating contribution types
* **Run this prompt using ollama, set temperature=0**
* **models=[gpt-oss-20b,mistral-small-3.2]**


---

### **Combined Dataset** (`combined_whole_dataset_per_comment.json.zip`)

* **12 043 comments**
* **6 912 introductory commits**
* **2 948 first-change commits**

Each record links a comment with its associated commits.

---


