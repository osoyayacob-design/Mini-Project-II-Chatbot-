"""
RAG Chatbot PPKD Jakarta Barat pakai LangChain + Groq + ChromaDB (lokal)

Script ini adalah hasil modifikasi dari materi dasar sesi LangChain:
- Blok 1: Model, Prompt Template, Output Parser
- Blok 2: LCEL (pipe, RunnableParallel, RunnablePassthrough)
- Blok 3: Data Ingestion (Document, Chunking, Embedding, Vector Store)
- Blok 4: RAG Chain (Context Injection, chain final)

Knowledge document di sini pakai 3 dokumen sumber tentang PPKD Jakarta
Barat, disimpan sebagai file .md di folder knowledge_docs/:
    - uu_final.md -> Pergub DKI No. 341/2016 (dasar hukum PPKD)
    - faq.md      -> FAQ resmi PPKD Jakarta Barat
    - jurusan.md  -> daftar kejuruan (observasi Instagram resmi)

Ketiga dokumen ini punya struktur Markdown yang BEDA-BEDA, jadi tidak
dipakai satu loader/splitter generik untuk semuanya (lihat komentar di
masing-masing fungsi proses_* di bagian 3).

Cara jalanin:
    python rag_chatbot.py

Prasyarat:
    - File .env berisi GROQ_API_KEY di folder yang sama dengan script ini
    - Folder knowledge_docs/ berisi uu_final.md, faq.md, dan jurusan.md
    - Package sudah terinstall (lihat requirements.txt)
"""

import os
import re

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

# ============================================================
# 1. KONFIGURASI
# ============================================================

CHAT_MODEL = "openai/gpt-oss-120b"
COLLECTION_NAME = "ppkd_jakarta_barat"
TOP_K = 5

# Dipakai HANYA untuk memecah chunk hasil split-by-header yang masih
# kepanjangan (lihat proses_pergub()). Bukan lagi splitter tunggal untuk
# semua dokumen seperti versi sebelumnya, karena faq.md dan jurusan.md
# displit dengan cara masing-masing yang menghasilkan chunk yang sudah
# pas ukurannya tanpa perlu dipotong lagi.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Folder berisi 3 dokumen sumber (.md). Nama file di bawah ini WAJIB
# cocok persis dengan nama file di folder knowledge_docs/.
KNOWLEDGE_DIR = "./knowledge_docs"
NAMA_FILE_PERGUB = "PerGub 341 tahun 2016.md"
NAMA_FILE_FAQ = "faq_ppkd_jakarta_barat.md"
NAMA_FILE_JURUSAN = "Jurusan.md"

# System prompt disimpan terpisah dari kode, supaya bisa diubah/di-review
# tanpa menyentuh logika program, dan supaya jejak revisinya jelas kalau
# dipakai bersama version control (git).
SYSTEM_PROMPT_PATH = "./system_prompt.md"


# ============================================================
# 2. SETUP MODEL
# ============================================================

def buat_model() -> ChatGroq:
    """Siapkan koneksi ke model chat lewat Groq."""
    return ChatGroq(
        model=CHAT_MODEL,
        temperature=0,
        reasoning_effort="low",
    )


# ============================================================
# 3. DATA INGESTION (Load -> Split -> Embed -> Store)
# ============================================================
# Tiga dokumen sumber punya struktur Markdown yang berbeda, jadi masing-
# masing perlu diproses dengan caranya sendiri (bukan satu splitter
# generik untuk semua):
#   - uu_final.md -> heading Markdown asli (## Bab, ### Pasal) yang rapi,
#                    jadi bisa split-by-header langsung.
#   - faq.md      -> heading cuma ada di level topik besar; isi tiap
#                    tanya-jawab pakai **bold**, BUKAN heading asli, jadi
#                    perlu di-parsing manual per pasangan Q&A.
#   - jurusan.md  -> heading (###) ADA tapi kosong isi (nama kejuruan itu
#                    sendiri adalah headingnya). Kalau displit apa adanya,
#                    isi tiap chunk jadi kosong dan tidak berguna untuk
#                    embedding, jadi perlu disusun ulang jadi kalimat.

def proses_pergub(path: str) -> list[Document]:
    """
    Split uu_final.md berdasarkan heading asli: Bab (##) lalu Pasal (###).
    Tiap potongan otomatis dapat metadata 'Bab' dan 'Pasal' dari teks
    headingnya sendiri, supaya nanti bisa disebutkan sebagai sumber
    jawaban yang presisi (bukan cuma "dari dokumen Pergub").
    """
    with open(path, encoding="utf-8") as f:
        teks = f.read()

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("##", "Bab"),
            ("###", "Pasal"),
        ],
        strip_headers=False,
    )
    potongan_awal = header_splitter.split_text(teks)

    # Beberapa pasal (terutama yang berisi daftar fungsi a-r) cukup
    # panjang. Split tahap kedua ini MENJAGA metadata Bab/Pasal yang
    # sudah didapat, cuma memecah isi yang kepanjangan jadi lebih kecil.
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    potongan_final = sub_splitter.split_documents(potongan_awal)

    for p in potongan_final:
        p.metadata["source"] = os.path.basename(path)

    return potongan_final


def proses_faq(path: str) -> list[Document]:
    """
    faq.md tidak pakai heading per pertanyaan, jadi MarkdownHeaderTextSplitter
    tidak bisa dipakai langsung. Pendekatannya: split manual pakai regex,
    1 chunk = 1 pasang Q&A, dengan metadata 'kategori' diambil dari heading
    besar (# FAQ - ...) terakhir yang dilewati sebelum pasangan Q&A itu.
    """
    with open(path, encoding="utf-8") as f:
        teks = f.read()

    kategori_saat_ini = "Umum"
    daftar_dokumen = []

    blok_qa = re.split(r"\n(?=Q:)", teks)

    for blok in blok_qa:
        cocok_kategori = re.findall(r"^##\s+(FAQ.*)$", blok, flags=re.MULTILINE)
        if cocok_kategori:
            kategori_saat_ini = cocok_kategori[-1].strip()

        cocok_qa = re.search(r"(Q:.*?)(?=\nQ:|\Z)", blok, flags=re.DOTALL)
        if not cocok_qa:
            continue

        isi_qa = cocok_qa.group(1).strip()
        if not isi_qa:
            continue

        daftar_dokumen.append(
            Document(
                page_content=isi_qa,
                metadata={
                    "source": os.path.basename(path),
                    "kategori": kategori_saat_ini,
                },
            )
        )

    return daftar_dokumen


def proses_jurusan(path: str) -> list[Document]:
    """
    Heading (###) di jurusan.md kosong isinya (nama kejuruan ITU SENDIRI
    adalah headingnya), jadi kalau displit apa adanya, page_content-nya
    kosong -> tidak berguna untuk embedding/retrieval.

    Solusinya: susun ulang jadi satu kalimat per kejuruan yang menyebutkan
    nama kejuruan + jalurnya (Reguler/MTU), supaya ada teks yang bisa
    di-embed, dan jalur tetap tercatat sebagai metadata terpisah.
    """
    with open(path, encoding="utf-8") as f:
        baris_baris = f.read().splitlines()

    jalur_saat_ini = None
    daftar_dokumen = []

    for baris in baris_baris:
        baris = baris.strip()

        if baris.startswith("## "):
            jalur_saat_ini = baris.replace("## ", "").strip()

        elif baris.startswith("### "):
            nama_kejuruan = baris.replace("### ", "").strip()
            if not nama_kejuruan or jalur_saat_ini is None:
                continue

            teks_chunk = f"Kejuruan: {nama_kejuruan}. Jalur: {jalur_saat_ini}."
            daftar_dokumen.append(
                Document(
                    page_content=teks_chunk,
                    metadata={
                        "source": os.path.basename(path),
                        "jalur": jalur_saat_ini,
                        "nama_kejuruan": nama_kejuruan,
                    },
                )
            )

    return daftar_dokumen


def muat_dokumen(folder: str) -> list[Document]:
    """
    Muat & proses ketiga dokumen sumber sekaligus. Bukan loader generik
    (seperti glob semua file lalu di-load satu cara), karena masing-
    masing dokumen butuh strategi split yang berbeda (lihat proses_*
    di atas).
    """
    semua_dokumen = []

    path_pergub = os.path.join(folder, NAMA_FILE_PERGUB)
    path_faq = os.path.join(folder, NAMA_FILE_FAQ)
    path_jurusan = os.path.join(folder, NAMA_FILE_JURUSAN)

    if not (os.path.exists(path_pergub) and os.path.exists(path_faq) and os.path.exists(path_jurusan)):
        raise FileNotFoundError(
            f"Salah satu dokumen sumber tidak ditemukan di '{folder}'. "
            f"Pastikan ada: {NAMA_FILE_PERGUB}, {NAMA_FILE_FAQ}, {NAMA_FILE_JURUSAN}."
        )

    semua_dokumen.extend(proses_pergub(path_pergub))
    semua_dokumen.extend(proses_faq(path_faq))
    semua_dokumen.extend(proses_jurusan(path_jurusan))

    return semua_dokumen


def bangun_vectorstore(dokumen: list) -> Chroma:
    """
    Ubah kumpulan Document jadi vector store siap dicari.

    Beda dari versi sebelumnya: TIDAK ADA splitter generik di sini lagi.
    Splitting sudah dilakukan di dalam muat_dokumen() (masing-masing
    dokumen dengan caranya sendiri), jadi di sini tinggal langsung embed
    dan simpan.
    """
    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
   
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
    )
    # reset_collection() dipakai supaya aman dijalankan berulang kali
    # (script ini bisa dijalankan ulang tanpa bikin data dobel atau error).
    vectorstore.reset_collection()
    vectorstore.add_documents(dokumen)

    return vectorstore


# ============================================================
# 4. RAG CHAIN (Context Injection -> Prompt -> Model -> Parser)
# ============================================================

def format_docs(daftar_dokumen: list[Document]) -> str:
    """Gabungkan beberapa chunk hasil retrieval jadi satu teks konteks."""
    return "\n\n".join(dok.page_content for dok in daftar_dokumen)

def muat_system_prompt(path: str) -> str:
    """
    Baca system prompt dari file .md terpisah.

    Dipisah dari kode supaya system prompt bisa direvisi (oleh siapa pun
    yang bertanggung jawab atas kualitas jawaban chatbot) tanpa perlu
    menyentuh atau memahami kode Python-nya sama sekali.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()

def buat_rag_chain(retriever, model: ChatGroq, system_prompt: str):
    """
    Satukan retriever, prompt, model, dan parser jadi satu chain LCEL.

    system_prompt (dari file system_prompt.md) berisi instruksi statis:
    peran asisten dan aturan anti-halusinasi. Placeholder {context} dan
    {question} sengaja TIDAK ditaruh di file itu, karena dua-duanya
    bukan instruksi tetap, melainkan bagian yang diisi ulang setiap kali
    ada pertanyaan baru, jadi tetap disusun di sini sebagai pesan "human".
    """
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Konteks:\n{context}\n\nPertanyaan: {question}"),
    ])

    rag_chain = (
        RunnableParallel(
            context=itemgetter("question") | retriever | format_docs,
            question=itemgetter("question"),
            chat_history=itemgetter("chat_history"),
        )
        | rag_prompt
        | model
        | StrOutputParser()
    )
    return rag_chain


# ============================================================
# 5. PROGRAM UTAMA
# ============================================================

def main():
    # load_dotenv() harus dipanggil sebelum ChatGroq dibuat, supaya
    # GROQ_API_KEY sudah ada di environment variable saat dibutuhkan.
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY tidak ditemukan. Pastikan file .env ada di folder "
            "yang sama dengan script ini dan berisi GROQ_API_KEY=..."
        )

    print("Menyiapkan model...")
    model = buat_model()

    print(f"Memuat dan memproses dokumen dari folder '{KNOWLEDGE_DIR}'...")
    dokumen = muat_dokumen(KNOWLEDGE_DIR)
    print(f"  -> {len(dokumen)} chunk berhasil dimuat dari 3 dokumen sumber.")

    print("Membangun vector store dari dokumen...")
    vectorstore = bangun_vectorstore(dokumen)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    print(f"Memuat system prompt dari '{SYSTEM_PROMPT_PATH}'...")
    system_prompt = muat_system_prompt(SYSTEM_PROMPT_PATH)

    print("Merakit RAG chain...")
    rag_chain = buat_rag_chain(retriever, model, system_prompt)

    print("\nRAG chatbot siap. Ketik pertanyaan, atau 'keluar' untuk berhenti.\n")

    riwayat = []

    while True:
        pertanyaan = input("Pertanyaan: ").strip()
        if pertanyaan.lower() in {"keluar", "exit", "quit", "selesai", "bye", "stop"}:
            print("Sampai jumpa.")
            break
        if not pertanyaan:
            continue

        jawaban = rag_chain.invoke({
        "question": pertanyaan,
        "chat_history": riwayat,
        })
        print(f"Jawaban : {jawaban}\n")

        riwayat.append(HumanMessage(content=pertanyaan))
        riwayat.append(AIMessage(content=jawaban))


if __name__ == "__main__":
    main()

# ============================================================
# KENAPA ADA "if __name__ == '__main__':" DI SINI?
# ============================================================
#
# Python punya "label" tersembunyi bernama __name__ di setiap file .py.
# Isinya beda tergantung cara file ini dipakai:
#
#   - Kalau file ini dijalankan LANGSUNG (python rag_chatbot.py),
#     maka __name__ otomatis berisi teks "__main__".
#
#   - Kalau file ini "dipinjam" fungsinya oleh file lain, misalnya
#     lewat "from rag_chatbot import muat_dokumen", maka __name__
#     berisi nama file ini sendiri, yaitu "rag_chatbot", BUKAN "__main__".
#
# Baris "if __name__ == '__main__':" ini jadi semacam pertanyaan yang
# ditanyakan Python ke dirinya sendiri: "Apakah saya sedang dijalankan
# langsung, atau cuma dipinjam file lain?"
#
# Kalau jawabannya "dijalankan langsung", baru main() dipanggil, dan
# seluruh program (load dokumen, bikin vectorstore, tanya-jawab) jalan.
#
# Kalau jawabannya "cuma dipinjam", main() TIDAK dipanggil. Ini penting
# supaya kalau suatu saat ada script lain yang cuma mau meminjam satu
# fungsi kecil dari sini (misalnya muat_dokumen() saja), seluruh RAG
# chatbot ini tidak ikut menyala tanpa diminta.
#
# Letaknya WAJIB di paling bawah file, setelah semua fungsi (def ...)
# selesai didefinisikan. Python membaca file dari atas ke bawah, jadi
# kalau baris ini ditaruh sebelum fungsi-fungsi yang dipanggil di dalam
# main() selesai "dikenalkan" ke Python, akan muncul error NameError.
# ============================================================