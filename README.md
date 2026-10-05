# 🤖 Asisten AI PPKD Jakarta Barat — RAG Chatbot

Chatbot berbasis **Retrieval-Augmented Generation (RAG)** yang menjawab pertanyaan seputar program pelatihan, syarat pendaftaran, dan aturan **Pusat Pelatihan Kerja Daerah (PPKD) Jakarta Barat**, berdasarkan dokumen yang sudah disediakan, bukan hasil karangan model.

> Mini Project 2 — Chatbot RAG dengan LangChain, Groq, dan ChromaDB

---

## 📌 Daftar Isi

- [Use Case & Latar Belakang](#-use-case--latar-belakang)
- [Fitur Utama](#-fitur-utama)
- [Arsitektur & Cara Kerja](#-arsitektur--cara-kerja)
- [Strategi Chunking per Dokumen](#-strategi-chunking-per-dokumen)
- [Pemilihan Model](#-pemilihan-model)
- [Format Output & Validasi](#-format-output--validasi)
- [Struktur Prompt / System Instruction](#-struktur-prompt--system-instruction)
- [Instalasi](#-instalasi)
- [Cara Menjalankan](#-cara-menjalankan)
- [Konfigurasi RAG](#-konfigurasi-rag)
- [Pengujian](#-pengujian)
- [Batasan Sistem](#-batasan-sistem)
- [Struktur Folder](#-struktur-folder)

---

## 🎯 Use Case & Latar Belakang

PPKD Jakarta Barat menyediakan pelatihan kerja bagi warga DKI Jakarta, tetapi informasinya tersebar di beberapa dokumen: dasar hukum, FAQ pendaftaran, dan daftar kejuruan. Calon peserta sering kesulitan menemukan jawaban yang cepat dan akurat.

Chatbot ini dibangun untuk menjawab pertanyaan seputar:

- Daftar kejuruan dan jalur pelatihan (Reguler dan MTU)
- Syarat dan cara pendaftaran
- Aturan dasar yang tercantum dalam Pergub
- Kanal resmi untuk informasi yang tidak tersedia di dokumen

Sumber jawaban **sepenuhnya berasal dari dokumen yang di-*load*** ke sistem. Chatbot dilarang menjawab dari pengetahuan umum, menebak jadwal, atau menjanjikan kepastian karier bagi alumni.

---

## ✨ Fitur Utama

- **RAG pipeline lengkap:** ingestion → chunking → embedding → retrieval → generation
- **Chunking khusus per dokumen:** tiga dokumen dengan struktur berbeda diproses dengan strategi berbeda
- **Anti-halusinasi:** jika informasi tidak ada, chatbot menjawab "Informasi tidak ditemukan di dokumen yang tersedia" dan mengarahkan ke kanal resmi
- **Sumber rujukan:** setiap jawaban di antarmuka dilengkapi daftar sumber (file, Bab/Pasal, kategori FAQ, atau kejuruan)
- **Memori percakapan:** riwayat chat ikut dikirim ke model sehingga pertanyaan lanjutan tetap nyambung
- **Antarmuka Streamlit:** jawaban tampil bertahap (*streaming*) dengan tombol "Mulai percakapan baru"
- **Mode CLI** (`rag_chatbot.py`) dan **skrip debug** (`debug_test.py`) untuk menguji retrieval

---

## 🏗️ Arsitektur & Cara Kerja

```
┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌──────────────┐
│ Dokumen Markdown │ → │ Chunking         │ → │ Embedding          │ → │ ChromaDB     │
│ (knowledge_docs/)│   │ (per dokumen)    │   │ (multilingual      │   │ (vector store│
│                  │   │                  │   │  MiniLM-L12)       │   │  lokal)      │
└──────────────────┘   └──────────────────┘   └────────────────────┘   └──────┬───────┘
                                                                              │
                                                                  Retrieval (top-k = 5)
                                                                              │
┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐          │
│ Jawaban          │ ← │ LLM (Groq,       │ ← │ System Prompt +    │ ←────────┘
│ + sumber rujukan │   │ gpt-oss-120b)    │   │ Konteks + Riwayat  │
└──────────────────┘   └──────────────────┘   └────────────────────┘
```

Rangkaian dibuat dengan **LCEL (LangChain Expression Language)**:

```python
RunnableParallel(
    context=itemgetter("question") | retriever | format_docs,
    question=itemgetter("question"),
    chat_history=itemgetter("chat_history"),
) | rag_prompt | model | StrOutputParser()
```

Vector store dibangun ulang setiap aplikasi dijalankan (`reset_collection()`), sehingga aman dijalankan berulang kali tanpa data ganda.

---

## ✂️ Strategi Chunking per Dokumen

Tiga dokumen sumber memiliki struktur Markdown yang berbeda, sehingga tidak memakai satu *splitter* generik.

| Dokumen | Masalah struktur | Strategi | Metadata |
|---|---|---|---|
| `PerGub 341 tahun 2016.md` | Heading Bab (`##`) dan Pasal (`###`) rapi | `MarkdownHeaderTextSplitter`, lalu `RecursiveCharacterTextSplitter` (800 karakter, overlap 100) untuk pasal yang panjang | `Bab`, `Pasal`, `source` |
| `faq_ppkd_jakarta_barat.md` | Tanya-jawab tidak memakai heading per pertanyaan | Split manual dengan regex, **1 chunk = 1 pasang Q&A** | `kategori`, `source` |
| `Jurusan.md` | Heading kejuruan tidak punya isi, sehingga chunk kosong | Disusun ulang menjadi kalimat: `Kejuruan: <nama>. Jalur: <jalur>.` | `jalur`, `nama_kejuruan`, `source` |

---

## 🧠 Pemilihan Model

| Komponen | Pilihan | Alasan |
|---|---|---|
| LLM | `openai/gpt-oss-120b` lewat **Groq** | Inferensi cepat, mendukung Bahasa Indonesia, dan cocok untuk tugas tanya-jawab berbasis konteks |
| Parameter LLM | `temperature=0`, `reasoning_effort="low"` | Jawaban konsisten, tidak kreatif berlebihan, dan respons lebih cepat |
| Embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Mendukung banyak bahasa termasuk Indonesia, ringan, dan berjalan **lokal** (tanpa biaya API embedding) |
| Vector store | **ChromaDB** (lokal) | Sederhana, tanpa server tambahan, cocok untuk skala dokumen kecil |
| Framework | **LangChain** | Menyediakan splitter, retriever, dan LCEL dalam satu ekosistem |

---

## ✅ Format Output & Validasi

Output model berupa **teks bebas** (`StrOutputParser`), bukan skema terstruktur. Validasi kualitas dilakukan lewat beberapa lapisan:

1. **System prompt ketat:** melarang pengetahuan di luar konteks, asumsi jadwal, dan opini pribadi
2. **Kalimat baku** untuk informasi yang tidak ditemukan, agar perilaku konsisten
3. **Sumber rujukan di UI:** pengguna dapat memeriksa chunk apa yang menjadi dasar jawaban
4. **Aturan "tidak menyempitkan fakta umum":** mencegah model mengaitkan aturan dengan kejuruan tertentu hanya karena kebetulan muncul bersamaan dalam konteks

---

## 📝 Struktur Prompt / System Instruction

System prompt disimpan terpisah di `system_prompt.md` agar bisa direvisi tanpa mengubah kode. Isinya:

| Bagian | Fungsi |
|---|---|
| Peran dan Identitas | Asisten pencari informasi, bukan staf atau representasi resmi PPKD, dan tidak boleh menampilkan logo |
| Sumber Kebenaran | Jawaban hanya dari konteks; dilarang mengarang detail (nomor pasal, tanggal, angka, durasi) |
| Tidak Menyempitkan Cakupan Fakta Umum | Tidak mengaitkan potongan konteks yang tidak berhubungan; tidak mengklaim daftar lengkap jika hanya menerima sebagian |
| Ketika Informasi Tidak Ditemukan | Kalimat baku, arahan ke Instagram, WhatsApp, atau email resmi, tanpa spekulasi |
| Pertanyaan di Luar Cakupan | Menolak opini pribadi, nasihat karier spesifik, dan topik di luar PPKD |
| Gaya Jawaban | Bahasa Indonesia ringkas, bullet untuk banyak poin, tidak memakai istilah teknis internal seperti "dokumen" atau "konteks" |

Pesan "human" disusun di kode: `Konteks: {context}` dan `Pertanyaan: {question}`, dengan `chat_history` di antaranya.

---

## ⚙️ Instalasi

**Prasyarat:** Python 3.10+ dan API key dari [Groq](https://console.groq.com/).

1. Clone repository:

   ```bash
   git clone https://github.com/osoyayacob-design/Mini-Project-II-Chatbot-.git
   cd Mini-Project-II-Chatbot-
   ```

2. Install dependensi:

   ```bash
   pip install -r requirements.txt
   pip install streamlit
   ```

   > `streamlit` dipakai oleh `app.py`. Jika belum tercantum di `requirements.txt`, install terpisah seperti di atas.

3. Buat file `.env` di folder utama:

   ```
   GROQ_API_KEY=isi_dengan_api_key_groq_kamu
   ```

4. Pastikan folder `knowledge_docs/` berisi tiga file berikut (nama harus persis sama):

   - `PerGub 341 tahun 2016.md`
   - `faq_ppkd_jakarta_barat.md`
   - `Jurusan.md`

> File `.env` jangan diunggah ke GitHub. Untuk Streamlit Cloud, isi `GROQ_API_KEY` di menu **Secrets**.

---

## ▶️ Cara Menjalankan

**Antarmuka web (Streamlit):**

```bash
streamlit run app.py
```

**Mode CLI (terminal):**

```bash
python rag_chatbot.py
```

Ketik pertanyaan lalu tekan Enter. Ketik `keluar` untuk berhenti.

Saat pertama dijalankan, model embedding akan diunduh sehingga proses awal bisa memakan waktu beberapa saat.

---

## 🔧 Konfigurasi RAG

Semua parameter ada di bagian konfigurasi `rag_chatbot.py`:

| Parameter | Nilai | Keterangan |
|---|---|---|
| `CHAT_MODEL` | `openai/gpt-oss-120b` | Model chat di Groq |
| `COLLECTION_NAME` | `ppkd_jakarta_barat` | Nama koleksi di ChromaDB |
| `TOP_K` | `5` | Jumlah chunk yang diambil per pertanyaan |
| `CHUNK_SIZE` | `800` | Ukuran maksimum chunk (khusus Pergub) |
| `CHUNK_OVERLAP` | `100` | Overlap antar chunk (khusus Pergub) |
| `KNOWLEDGE_DIR` | `./knowledge_docs` | Folder dokumen sumber |
| `SYSTEM_PROMPT_PATH` | `./system_prompt.md` | Lokasi system prompt |

---

## 🧪 Pengujian

`debug_test.py` dipakai untuk memeriksa **kualitas retrieval** tanpa memanggil LLM. Skrip memuat dokumen, membangun vector store, lalu mencetak potongan chunk yang diambil untuk satu pertanyaan uji, contohnya `"KTPnya harus DKI ya?"`.

```bash
python debug_test.py
```

Cara ini membantu memastikan chunk yang relevan benar-benar terambil sebelum menilai jawaban model. Pengujian jawaban lengkap dilakukan lewat antarmuka Streamlit atau mode CLI.

---

## ⚠️ Batasan Sistem

- Jawaban hanya setara dengan isi dokumen sumber; data bisa tidak mencerminkan kondisi terkini (jadwal, batch, dan ketentuan yang berubah).
- Informasi jadwal pendaftaran tidak tersedia, sehingga pengguna diarahkan ke Instagram `@ppkd_jakarta_barat`.
- Dokumen `Jurusan.md` hanya memuat nama kejuruan dan jalur, tanpa detail modul atau durasi.
- Bukan sumber hukum resmi dan bukan pengganti konsultasi dengan pihak PPKD.
- Retrieval memakai top-k tetap, sehingga pertanyaan yang butuh daftar lengkap bisa hanya mendapat sebagian data.
- Vector store dibangun ulang setiap aplikasi dijalankan, sehingga startup awal lebih lambat.
- Membutuhkan koneksi internet untuk memanggil API Groq.

---

## 📁 Struktur Folder

```
.
├── .streamlit/            # Konfigurasi Streamlit
├── knowledge_docs/        # Dokumen sumber (Markdown)
│   ├── PerGub 341 tahun 2016.md
│   ├── faq_ppkd_jakarta_barat.md
│   └── Jurusan.md
├── app.py                 # Antarmuka web Streamlit
├── rag_chatbot.py         # Logika RAG: ingestion, retriever, chain, mode CLI
├── debug_test.py          # Skrip uji retrieval
├── system_prompt.md       # System prompt chatbot
├── requirements.txt       # Daftar dependensi
├── .gitignore
└── README.md
```

---

## 👤 Penulis

**Yacob**
