from rag_chatbot import KNOWLEDGE_DIR, TOP_K, muat_dokumen, bangun_vectorstore

dokumen = muat_dokumen(KNOWLEDGE_DIR)
vectorstore = bangun_vectorstore(dokumen)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

hasil = retriever.invoke("KTPnya harus DKI ya?")
for dok in hasil:
    print("---")
    print(dok.page_content[:200])