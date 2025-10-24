
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
├──Dataset
│   └── combined_whole_dataset_per_comment.json.zip
│       # 12,043 comments + 6,912 intro commits + 2,948 first-change commits
|
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
* **`dawid_skene.py`:** Performs Dawid–Skene EM aggregation with human labels as gold.

---

### **Combined Dataset** (`combined_whole_dataset_per_comment.json.zip`)

* **12 043 comments**
* **6 912 introductory commits**
* **2 948 first-change commits**

Each record links a comment with its associated commits.

---

## 📜 Citation

If you use this dataset, please cite:

```bibtex
@dataset{oss_llm_annotations_2025,
  title   = {LLM-Assisted Code: Annotations & Commits Dataset},
  author  = {Al Mujahid, Abdullah and Collaborators},
  year    = {2025},
  note    = {Human and LLM annotations of code comments with linked commit metadata},
  url     = {<repo_url>}
}
```

---

