### TitleGraph

#### Install
`pip install title-graph`

#### Build
```python setup.py sdist```

```twine upload dist/*```

#### Usage
```python
from title_graph import TitleGraph

tg = TitleGraph()

# Find likely future job titles
>>> tg.query_forward('graduate student', topn=3)
('software engineering intern', 0.8774160146713257), ('software engineer intern', 0.8586706519126892), ('software development engineering intern', 0.8407694101333618)]

# Find likely previous job titles
tg.query_backwards('principal engineer', topn=3)
[('engineer', 0.8578487634658813), ('dev engineer', 0.7771068811416626), ('sr engineer', 0.7421303987503052)]

# Semantic search job titles
tg.query_similar_semantic('front-end ninja', topn=3)
[('frontend developer', 0.7447044849395752), ('frontend engineer', 0.6994348764419556), ('front end engineer', 0.6821744441986084)]
```