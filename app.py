import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from rag_chatbot import (
    KNOWLEDGE_DIR,
    SYSTEM_PROMPT_PATH,
    TOP_K,
    buat_model,
    muat_dokumen,
    bangun_vectorstore,
    muat_system_prompt,
    buat_rag_chain,
)


# ============================================================
# 1. PENGATURAN HALAMAN
# ============================================================

st.set_page_config(
    page_title="Asisten PPKD Jakarta Barat",
    page_icon=":material/school:",
)

st.markdown(
    """
    <style>
    /* ---------- Layout dasar ---------- */
    .stMainBlockContainer {
        max-width: 820px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
    }

    /* ---------- Judul & subtitle ---------- */
    h1 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #1A2FE0 !important;
        margin-bottom: 0.25rem !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stCaptionContainer"] {
        color: #6B7280 !important;
        font-size: 0.875rem !important;
    }

    /* ---------- Garis aksen biru tipis di bawah header ---------- */
    h1 + div[data-testid="stCaptionContainer"] {
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #1A2FE0;
        margin-bottom: 1rem;
    }

    /* ---------- Bubble chat ---------- */
    [data-testid="stChatMessage"] {
        background-color: transparent;
        padding: 1rem 0;
        border-bottom: 1px solid #F0F1F3;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: #F7F8FA;
        border-right: 1px solid #ECEDEF;
    }
    [data-testid="stSidebar"] h2 {
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #1A2FE0 !important;
    }

    /* ---------- Tombol ---------- */
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid #E0E2E6 !important;
        font-size: 0.875rem !important;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #1A2FE0 !important;
        color: #1A2FE0 !important;
        background-color: #F0F3FF !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #1A2FE0 !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1425B8 !important;
        color: #FFFFFF !important;
    }

    /* ---------- Expander sumber jawaban ---------- */
    [data-testid="stExpander"] {
        border: 1px solid #ECEDEF !important;
        border-radius: 8px !important;
        background-color: #FAFBFC !important;
        margin-top: 0.5rem;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.85rem !important;
        color: #1A2FE0 !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] p {
        font-family: inherit !important;
        font-size: 0.85rem !important;
    }

    /* ---------- Input chat ---------- */
    [data-testid="stChatInput"] textarea {
        border-radius: 10px !important;
        border: 1px solid #E0E2E6 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #1A2FE0 !important;
        box-shadow: 0 0 0 2px rgba(26, 47, 224, 0.12) !important;
    }

    /* ---------- Bersihkan elemen Streamlit ---------- */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. CEK API KEY
# ============================================================

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY belum diisi. Cek file .env (di laptop) "
        "atau menu Secrets (di Streamlit Cloud)."
    )
    st.stop()


# ============================================================
# 3. SIAPKAN MESIN CHATBOT (sekali saja, lalu disimpan)
# ============================================================

@st.cache_resource(show_spinner="Menyiapkan chatbot, mohon tunggu sebentar...")
def siapkan_chatbot():
    model = buat_model()
    dokumen = muat_dokumen(KNOWLEDGE_DIR)
    vectorstore = bangun_vectorstore(dokumen)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    system_prompt = muat_system_prompt(SYSTEM_PROMPT_PATH)
    rag_chain = buat_rag_chain(retriever, model, system_prompt)
    return rag_chain, retriever


rag_chain, retriever = siapkan_chatbot()


# ============================================================
# 4. BUKU CATATAN PERCAKAPAN
# ============================================================

if "riwayat" not in st.session_state:
    st.session_state.riwayat = []

if "riwayat_chain" not in st.session_state:
    st.session_state.riwayat_chain = []


# ============================================================
# 5. TAMPILAN
# ============================================================

with st.sidebar:
    st.header("Tentang chatbot ini")
    st.write(
        "Chatbot ini menjawab pertanyaan seputar PPKD Jakarta Barat "
        "(aturan dasar, FAQ pendaftaran, dan daftar jurusan) berdasarkan "
        "dokumen resmi yang tersedia."
    )
    st.caption("Jawaban hanya diambil dari dokumen sumber, bukan dari internet.")
    st.caption(
        "Data mungkin tidak selalu mencerminkan kondisi terkini. "
        "Untuk info jadwal/batch terbaru, cek Instagram @ppkd_jakarta_barat."
    )
    if st.button("Mulai percakapan baru", type="primary"):
        st.session_state.riwayat = []
        st.session_state.riwayat_chain = []

st.title("Asisten Informasi PPKD Jakarta Barat")
st.caption(
    "Tanyakan seputar pendaftaran, syarat, dan jenis pelatihan PPKD Jakarta Barat."
)

with st.chat_message("assistant"):
    st.markdown(
        "Halo, silakan ajukan pertanyaan. Contoh: "
        "Apa saja kejuruan yang tersedia di jalur Reguler?"
    )

for pesan in st.session_state.riwayat:
    with st.chat_message(pesan["role"]):
        st.markdown(pesan["isi"])
        if pesan.get("sumber"):
            with st.expander("Lihat sumber jawaban"):
                st.markdown(pesan["sumber"])


# ============================================================
# 6. TANYA JAWAB
# ============================================================

def format_sumber(dokumen_terambil: list) -> str:
    sumber_unik = []
    for dok in dokumen_terambil:
        nama_file = dok.metadata.get("source", "tidak diketahui")

        if "Pasal" in dok.metadata:
            keterangan = f"{nama_file} — {dok.metadata.get('Bab', '')}, {dok.metadata['Pasal']}"
        elif "kategori" in dok.metadata:
            keterangan = f"{nama_file} — {dok.metadata['kategori']}"
        elif "nama_kejuruan" in dok.metadata:
            keterangan = f"{nama_file} — {dok.metadata['nama_kejuruan']} ({dok.metadata['jalur']})"
        else:
            keterangan = nama_file

        if keterangan not in sumber_unik:
            sumber_unik.append(keterangan)

    return "\n".join(f"- {s}" for s in sumber_unik)


pertanyaan = st.chat_input("Tulis pertanyaan Anda di sini...")

if pertanyaan:
    with st.chat_message("user"):
        st.markdown(pertanyaan)
    st.session_state.riwayat.append({"role": "user", "isi": pertanyaan, "sumber": None})

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban di dokumen..."):
            dokumen_terambil = retriever.invoke(pertanyaan)
            jawaban = st.write_stream(
                rag_chain.stream({
                    "question": pertanyaan,
                    "chat_history": st.session_state.riwayat_chain,
                })
            )

        teks_sumber = format_sumber(dokumen_terambil)
        if teks_sumber:
            with st.expander("Lihat sumber jawaban"):
                st.markdown(teks_sumber)

    st.session_state.riwayat.append(
        {"role": "assistant", "isi": jawaban, "sumber": teks_sumber}
    )

    st.session_state.riwayat_chain.append(HumanMessage(content=pertanyaan))
    st.session_state.riwayat_chain.append(AIMessage(content=jawaban))