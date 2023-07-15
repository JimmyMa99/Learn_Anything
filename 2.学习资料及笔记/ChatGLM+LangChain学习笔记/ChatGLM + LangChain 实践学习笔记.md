# ChatGLM + LangChain：基于本地知识库的 ChatGLM 等大语言模型应用实现

[【官方教程】ChatGLM + LangChain 实践培训_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV13M4y1e7cN/)

# 1.文档问答

## 1.1基于单一文档的问答实现原理

1. 加载本地文档（读取本地文档，加载为文本）
2. 文本拆分（将文本按字符、长度或者语义进行拆分）
3. 根据提问匹配文本（根据用户提问对文本进行字符匹配或语义检索）
4. 构建prompt（将匹配文本，用户提问加入prompt模板）
5. LLM生成问答（将prompt发送给LLM获得基于文档内容的回答）

## 1.2基于本地知识库问答的实现原理

![Untitled](ChatGLM%20+%20LangChain%EF%BC%9A%E5%9F%BA%E4%BA%8E%E6%9C%AC%E5%9C%B0%E7%9F%A5%E8%AF%86%E5%BA%93%E7%9A%84%20ChatGLM%20%E7%AD%89%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E5%AE%9E%E7%8E%B0%20ec64909dabcb498ab063a747c424dea1/Untitled.png)

A.对本地问答进行向量化

1. Unstructured Loader：知识库文档加载 支持.txt .md .pdf等常见文本格式抓成text
2. Text Splitter：将text向量切分形成文本段落
3. Text Chunks：将文本段落利用embedding模型向量化
4. VectorStore：本地知识库本体 存储了语义向量（到此已经将知识库向量化了）

B.对用户提问Query的向量化

1. 与A中相同的embedding处理
2. 生成Query向量

C.针对VectorStore和QueryVector进行文本匹配

1. 计算向量相似度
2. 形成相关文段
3. 生成提示词模板
4. 生成promp

# 2.个人助理

# 3.查询表格数据

# 4.与API交互

# 5.信息提取

# 6.文档总结

# 7.文档总结