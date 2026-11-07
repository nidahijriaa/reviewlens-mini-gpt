# PROJECT_CONTEXT — ReviewLens GPT

**Indonesian Customer Review Insight Generator dengan Mini-GPT from scratch**

Tugas mata kuliah Generative AI · Tenggat: **Senin, 9 November 2026**
Status dokumen: **Revisi 2 — siap ditinjau untuk persetujuan tim**
Dokumen pasangan: [INTERFACES.md](INTERFACES.md) (kontrak antar-modul).
Jika ada perbedaan detail teknis, INTERFACES.md yang berlaku.

---

## 1. Gambaran Umum

ReviewLens GPT adalah aplikasi yang membaca ulasan pelanggan berbahasa Indonesia
lalu menghasilkan tiga jenis insight:

| Insight | Pertanyaan yang dijawab | Contoh keluaran |
|---|---|---|
| **Positive Insights** | Apa yang disukai pelanggan? | "pengiriman cepat", "barang sesuai deskripsi" |
| **Customer Pain Points** | Apa yang dikeluhkan? | "produk rusak atau cacat", "pengiriman lambat" |
| **Improvement Suggestions** | Apa yang sebaiknya diperbaiki? | "perketat quality control sebelum pengiriman" |

Teks insight **dihasilkan oleh model Mini-GPT (decoder-only Transformer) yang
dilatih dari nol** dengan objective next-token prediction. Proyek ini tidak memakai
pretrained model maupun API LLM eksternal.

## 2. Tujuan

1. Mengimplementasikan decoder-only Transformer secara manual di PyTorch:
   multi-head self-attention, causal mask, learned positional embedding,
   blok pre-LayerNorm, dan inisialisasi bobot acak.
2. Melatih model dari nol pada ulasan e-commerce berbahasa Indonesia.
3. Membangun aplikasi demo (Streamlit) yang menerima satu atau banyak ulasan
   dan menampilkan ketiga insight.
4. Menyediakan evaluasi kuantitatif sederhana untuk presentasi.

## 3. Ruang Lingkup

**Termasuk (MVP):**
- Preprocessing dataset ulasan (Lazada Indonesia, kategori elektronik).
- Pembuatan *silver label* insight berbasis aturan (lexicon aspek) sebagai
  target training (lihat §6).
- Tokenizer word-level yang dibangun dari korpus sendiri.
- Mini-GPT, training loop dengan checkpoint yang bisa dilanjutkan (resumable).
- Inferensi per ulasan dan agregasi multi-ulasan (frekuensi insight).
- Evaluasi: test perplexity, format validity, aspect F1, exact match.
- Aplikasi Streamlit: input teks atau upload CSV, lalu tiga panel insight.
- Unit test komponen inti.

**Di luar lingkup** (tidak dikerjakan sebelum MVP selesai):
- Multi-GPU, `DataParallel`, distributed training, mixed precision.
- Tokenizer BPE dan KV-cache.
- Anotasi manual skala besar, deployment ke cloud, autentikasi.
- Bahasa selain Indonesia dan domain selain ulasan produk.

## 4. Analisis Repository & Data (per 7 Nov 2026)

### 4.1 Repository `reviewlens-mini-gpt`
- Branch `main` belum memiliki commit. Saat ini isinya hanya dokumen perencanaan.
- Remote: `origin → https://github.com/nidahijriaa/reviewlens-mini-gpt.git`.
- Konsekuensi: struktur folder dirancang dari nol (lihat §7).

### 4.2 Dataset (di luar repo: `../Dataset Costumer Review Indonesia/`)

| File | Baris | Ukuran | Dipakai di pipeline? |
|---|---|---|---|
| `Processed Dataset.csv` | 41.711 | 7 MB | **Ya, sumber utama** |
| `Raw Dataset.csv` | 203.786 | 30 MB | Tidak (hanya referensi kategori produk) |

Dataset diperlakukan **read-only**. Kode hanya membaca file ini dan tidak
pernah menulis ulang atau mengubah kolom aslinya.

**Pemetaan kolom.** Nama kolom asli dipertahankan di file. Penggantian nama
hanya dilakukan di memori oleh `load_reviews`:

| Kolom asli di CSV | Nama internal | Keterangan |
|---|---|---|
| `reviewContent` | `review` | teks ulasan (sudah lowercase, tanpa tanda baca) |
| `sentiment_label` | `sentiment` | `positive` / `negative` / `neutral`, label lemah |
| `rating` | `rating` | 1.0–5.0; 113 baris kosong (NaN diizinkan) |
| `itemId` | `item_id` | ID produk |
| `confidence` | (dibaca, tidak dipakai di MVP) | skor keyakinan `sentiment_label` |

Kolom lain (`name`, `reviewTitle`, `relevanceScore`, `boughtDate`,
`clientType`, `retrievedDate`) diabaikan.

Statistik `Processed Dataset.csv`:
- Sentimen: positive 28.574 (68%), negative 9.049 (22%), neutral 4.088 (10%).
- Rating: 5★ 32.267, 4★ 4.172, 3★ 1.675, 2★ 817, 1★ 2.667, kosong 113.
- Panjang ulasan: median **8 kata**, p90 25 kata, maksimum 361 kata.
- 3.288 produk unik. Kategori (dari raw): harddisk eksternal, flash drive,
  smart TV, televisi digital, laptop.

Catatan:
- `sentiment_label` tampaknya merupakan hasil model lain (ada kolom
  `confidence`). Karena itu label ini diperlakukan sebagai label lemah, bukan
  ground truth.
- Banyak slang dan typo ("mantaaapppp", "sesui", "gak"), sehingga perlu
  normalisasi ringan.
- Data tidak seimbang (didominasi ulasan positif). Solusinya: stratified split
  dan oversampling ringan ulasan negatif pada split train.
- **Tidak ada label insight** di dataset. Inilah alasan perlunya silver label (§6).

## 5. Persyaratan Teknis & Pemetaan ke Desain

| Persyaratan | Cara dipenuhi |
|---|---|
| Python + PyTorch | Seluruh kode ditulis dalam Python; model berupa `torch.nn.Module` |
| Decoder-only Mini-Transformer | `MiniGPT`: token emb + pos emb → N × `TransformerBlock` → LayerNorm → LM head |
| Multi-Head Self-Attention manual | `MultiHeadSelfAttention` menghitung Q, K, V dan `softmax(QKᵀ/√d)·V` sendiri; **tidak** memakai `nn.MultiheadAttention` / `F.scaled_dot_product_attention` |
| Causal masking | Buffer matriks segitiga bawah; posisi masa depan diisi `-inf` sebelum softmax; dibuktikan dengan unit test |
| Next-token prediction | Loss `cross_entropy(logits(x), y)` dengan `x = ids[:-1]`, `y = ids[1:]` |
| Random init | `_init_weights`: Normal(0, 0.02), bias 0; tanpa memuat bobot eksternal |
| Tanpa pretrained LM | Tokenizer dan embedding dibangun dari korpus sendiri |
| Tanpa API LLM | Inferensi 100% lokal dari checkpoint sendiri |

## 6. Pendekatan Inti: Bagaimana Model "Menghasilkan Insight"

Masalahnya, dataset hanya berisi ulasan dan sentimen, tanpa contoh insight.
Solusi yang realistis untuk tenggat 2 hari:

1. **Silver labeling berbasis aturan** (`labeling.py`). Lexicon berisi 9 aspek:
   kualitas produk, fungsi/kerusakan, pengiriman, kemasan, kesesuaian
   deskripsi, harga, pelayanan penjual, keaslian, dan garansi. Setiap aspek
   punya kata kunci, cue polaritas, dan frasa baku untuk positif, keluhan, dan
   saran. Jika cue polaritas tidak ditemukan, `sentiment` dipakai sebagai
   fallback.
2. **Penanganan negasi yang konsisten.** Normalisasi dilakukan satu kali di
   `clean_text`: `gak`, `ga` (beserta varian `nggak`, `gk`, `tdk`) diubah
   menjadi `tidak`. Setelah itu labeling hanya mengenal empat kata negasi
   kanonik: `tidak`, `kurang`, `belum`, `bukan`. Detail ada di INTERFACES.md §4–5.
3. **Format sekuens training** (satu ulasan = satu sekuens):
   ```
   <bos> <review> barang bagus tapi pengiriman lama sekali
   <pos> kualitas produk bagus
   <neg> pengiriman lambat
   <sug> percepat proses pengiriman <eos>
   ```
   Bagian yang kosong diisi token `<none>`.
4. **Training.** Mini-GPT mempelajari pola bahasa ulasan *dan* pemetaan
   ulasan → insight murni lewat next-token prediction.
5. **Inferensi.** Prompt `<bos> <review> … <pos>` diberikan ke model, lalu model
   meng-generate sisa sekuens secara autoregresif. Parser kemudian memecah
   hasilnya menjadi tiga bagian.
6. **Agregasi multi-ulasan.** Hasil per ulasan dihitung frekuensinya untuk
   menghasilkan top-k positive insights, pain points, dan saran beserta
   persentasenya.

Nilai jual untuk presentasi: aturan hanya dipakai untuk membuat target pada
data training. Model belajar menggeneralisasi ke ulasan dengan slang, typo,
atau kombinasi baru yang tidak tertangkap lexicon. Hal ini dapat ditunjukkan
lewat evaluasi dan contoh demo. Keterbatasan pendekatan ini dinyatakan secara
jujur di laporan.

## 7. Struktur Repository yang Diusulkan

```
reviewlens-mini-gpt/
├── PROJECT_CONTEXT.md        # dokumen ini
├── INTERFACES.md             # kontrak antar-modul
├── README.md                 # cara install, train (Kaggle/Colab), run demo
├── requirements.txt
├── .gitignore                # data/, checkpoints/, __pycache__/, .venv/
├── notebooks/
│   └── train_kaggle.ipynb    # clone repo → prepare_data → train → evaluate
├── data/                     # gitignored; path bisa diatur via CLI
│   ├── raw/                  # tempat CSV untuk pengembangan lokal
│   └── processed/            # train/val/test.jsonl + vocab.json
├── checkpoints/              # gitignored: best.pt, last.pt, history.json, vocab.json, eval.json
├── src/reviewlens/
│   ├── __init__.py
│   ├── config.py             # ModelConfig, TrainConfig, Paths, get_device()
│   ├── preprocess.py         # load CSV + mapping kolom, cleaning, normalisasi, split
│   ├── labeling.py           # lexicon aspek → silver insight
│   ├── tokenizer.py          # WordTokenizer + token khusus
│   ├── dataset.py            # format sekuens, truncation, Dataset/collate
│   ├── attention.py          # MultiHeadSelfAttention (manual + causal mask)
│   ├── model.py              # FeedForward, TransformerBlock, MiniGPT, generate
│   ├── train.py              # training loop, checkpoint, resume (CLI)
│   ├── inference.py          # InsightGenerator: per ulasan + agregasi
│   └── evaluate.py           # test_perplexity, format_validity, aspect_f1, exact_match (CLI)
├── app/
│   └── streamlit_app.py      # UI demo
├── scripts/
│   └── prepare_data.py       # pipeline data end-to-end (CLI)
└── tests/
    ├── test_preprocess.py
    ├── test_tokenizer.py
    ├── test_labeling.py
    ├── test_dataset.py       # batas panjang sekuens
    ├── test_attention.py     # shape + causal mask tidak bocor
    ├── test_model.py         # forward, loss, overfit 1 batch, generate
    ├── test_checkpoint.py    # save → resume melanjutkan epoch/step
    └── test_inference.py     # parser & agregasi
```

Struktur sengaja dibuat datar (satu paket, satu file per tanggung jawab) agar
mudah dibaca dosen dan cepat dikerjakan.

## 8. Arsitektur Model & Estimasi Ukuran

### 8.1 Hyperparameter awal

| Hyperparameter | Nilai | Alasan |
|---|---|---|
| Tokenizer | word-level, `min_freq=3`, vocab 6.000–10.000 | ulasan pendek; sekuens jauh lebih pendek dibanding char-level |
| `block_size` | 128 | panjang input maksimum model (lihat §8.3) |
| `n_layer` | 4 | |
| `n_head` | 4 | `d_head = 64` |
| `d_model` | 256 | |
| `d_ff` | 1024 (4×) | GELU |
| `dropout` | 0.1 | |
| Positional embedding | learned, `(block_size, d_model)` | |
| Weight tying | token embedding ↔ LM head (LM head tanpa bias) | |
| Optimizer | AdamW, lr 3e-4, warmup linear + cosine decay, grad clip 1.0 | |
| Batch | 64 | |
| Epoch | 5–8, simpan best berdasarkan val loss | |
| Decoding | greedy (default demo, deterministik), opsi temperature/top-k | |

Pre-LayerNorm dipilih karena lebih stabil untuk training singkat.

### 8.2 Estimasi jumlah parameter

| Komponen | Rumus | Parameter |
|---|---|---|
| Token embedding (tied dengan LM head) | V × 256 | 1.536.000 – 2.560.000 |
| Positional embedding | 128 × 256 | 32.768 |
| Per blok: attention QKV | 256×768 + 768 | 197.376 |
| Per blok: attention output proj | 256×256 + 256 | 65.792 |
| Per blok: FFN | (256×1024 + 1024) + (1024×256 + 256) | 525.568 |
| Per blok: 2 × LayerNorm | 4 × 256 | 1.024 |
| **Subtotal per blok** | | **789.760** |
| 4 blok | | 3.159.040 |
| LayerNorm akhir | 2 × 256 | 512 |
| LM head | tied, tanpa bias | 0 |

| vocab_size | Total parameter |
|---|---|
| 6.000 | **≈ 4,73 juta** (4.728.320) |
| 8.000 | ≈ 5,24 juta (5.240.320) |
| 10.000 | **≈ 5,75 juta** (5.752.320) |

**Estimasi resmi: ≈ 4,7–5,8 juta parameter.** Angka pastinya dilaporkan oleh
`MiniGPT.num_parameters()` setelah vocab terbentuk.

### 8.3 Aturan panjang sekuens

- `block_size` (128) adalah **panjang input maksimum** yang diterima model.
  Ukuran ini sama dengan ukuran positional embedding dan causal mask.
- Satu sekuens training lengkap berisi **paling banyak `block_size + 1` = 129 token**.
- Setelah digeser, `x = ids[:-1]` dan `y = ids[1:]` memiliki panjang yang sama,
  yaitu paling banyak `block_size`. `x` tidak pernah melebihi `block_size`.
- Bagian target insight (dari `<pos>` sampai `<eos>`) dibatasi
  `MAX_TARGET_TOKENS = 48`. Bagian yang dipotong adalah **teks ulasan**, bukan
  target: `MAX_REVIEW_TOKENS = block_size + 1 − 2 − 48 = 79` kata (mencakup
  >98% ulasan, karena p90 = 25 kata).
- Batas ulasan yang sama dipakai saat inferensi, sehingga prompt ditambah token
  hasil generate tidak pernah melebihi `block_size` (perhitungan di
  INTERFACES.md §7).

## 9. Lingkungan Pengembangan & Training

| Peran | Tools |
|---|---|
| Coding / editor | Google Antigravity |
| Version control | GitHub (`nidahijriaa/reviewlens-mini-gpt`) |
| **Training utama** | **Kaggle Notebooks**, GPU gratis (T4 ×2 jika tersedia) |
| Training cadangan | Google Colab Free (GPU T4) |
| Demo aplikasi & unit test | Laptop lokal (CPU cukup untuk inferensi model ±5 juta parameter) |

### 9.1 Kebijakan device
- Pemilihan device otomatis: `device = "cuda" if torch.cuda.is_available() else "cpu"`.
- `"cuda"` berarti **satu GPU, `cuda:0`**. Meskipun Kaggle menyediakan T4 ×2,
  GPU kedua tidak dipakai. Proyek ini tidak memakai `DataParallel`,
  `DistributedDataParallel`, atau multi-GPU.
- CPU hanya menjadi fallback (pengembangan lokal, unit test, inferensi demo).
- Checkpoint dimuat dengan `map_location=device`, sehingga model yang dilatih di
  GPU dapat dijalankan di CPU.

### 9.2 Alur kerja Kaggle
1. Upload `Processed Dataset.csv` sebagai **Kaggle Dataset privat** (tanpa
   diubah). File akan tersedia secara read-only di `/kaggle/input/<slug>/`.
2. Di notebook: `git clone` repo → `pip install` paket yang belum ada →
   `prepare_data.py --csv /kaggle/input/<slug>/"Processed Dataset.csv" --out-dir /kaggle/working/data/processed`
   → `python -m reviewlens.train --data-dir /kaggle/working/data/processed --checkpoint-dir /kaggle/working/checkpoints`.
3. **Penting: file di `/kaggle/working` hilang saat sesi berakhir** kecuali
   notebook disimpan dengan *Save Version* ("Save & Run All" / "Quick Save"),
   sehingga isinya menjadi *notebook output*. Setelah training (dan sebaiknya
   juga di tengah training panjang):
   - simpan versi notebook agar `checkpoints/` tercatat sebagai output, **dan/atau**
   - unduh `best.pt`, `last.pt`, `history.json`, dan `vocab.json` secara manual.
4. **Melanjutkan training di sesi baru:** tambahkan output notebook sebelumnya
   sebagai input (*Add Input → Notebook Output*), salin `last.pt` dan `vocab.json`
   ke `/kaggle/working/checkpoints/`, lalu jalankan `train` lagi. Training
   otomatis dilanjutkan dari epoch berikutnya (lihat §9.3).
5. Colab (cadangan): mount Google Drive dan arahkan `--checkpoint-dir` ke folder
   di Drive agar checkpoint tidak hilang ketika runtime terputus.

### 9.3 Checkpoint & resumable training

| File | Kapan disimpan | Kegunaan |
|---|---|---|
| `checkpoints/last.pt` | setiap akhir epoch | melanjutkan training |
| `checkpoints/best.pt` | setiap kali val loss membaik | inferensi, evaluasi, demo |
| `checkpoints/history.json` | setiap akhir epoch | kurva loss untuk laporan |
| `checkpoints/vocab.json` | sekali di awal training (salinan) | agar checkpoint bisa dipakai di mesin lain |

Isi setiap file `.pt`: `model_state`, `optimizer_state`, `model_cfg`, `epoch`,
`step`, `best_val_loss`, `vocab_path`.

- Jika `last.pt` yang valid ada di `--checkpoint-dir`, training **otomatis
  dilanjutkan** dari `epoch + 1`, tidak dimulai lagi dari epoch 1. Flag
  `--no-resume` memaksa training dari awal.
- Learning rate dihitung sebagai fungsi murni dari `step`. Karena itu state
  scheduler tidak perlu disimpan, dan resume tetap menghasilkan jadwal LR yang
  benar.
- Kontrak lengkap ada di INTERFACES.md §9.

### 9.4 Dependency

| Paket | Versi minimum | Fungsi | Kaggle/Colab | Lokal |
|---|---|---|---|---|
| Python | 3.10+ | | ✔ bawaan | 3.12.10 ✔ |
| torch | 2.1+ | model & training | ✔ bawaan (CUDA) | 2.14.1 ✔ (CPU) |
| numpy | 1.24+ | util numerik | ✔ bawaan | ✔ |
| pandas | 2.0+ | baca/olah CSV | ✔ bawaan | ✘ perlu install |
| tqdm | 4.65+ | progress bar | ✔ bawaan | ✘ |
| matplotlib | 3.7+ | kurva loss | ✔ bawaan | ✔ |
| pytest | 7.0+ | unit test | opsional | ✘ |
| streamlit | 1.30+ | UI demo | tidak perlu | ✘ |

Tidak diperlukan: `transformers`, `tokenizers`, `sentencepiece`, `accelerate`,
atau API key apa pun. Instalasi dilakukan setelah rencana disetujui.

Setup lokal (untuk test & demo):
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```
Reprodusibilitas: seed tetap (`random`, `numpy`, `torch`, `torch.cuda`).

### 9.5 Estimasi waktu training (diverifikasi di F4)
Sekitar 40 ribu sekuens berukuran ≤128 token dengan batch 64 menghasilkan
±650 step per epoch. Di T4, model ±5 juta parameter diperkirakan butuh < 1 menit
per epoch. Di CPU, diperkirakan beberapa menit per epoch. Training penuh
diperkirakan < 15 menit di Kaggle.

## 10. Batasan Implementasi (aturan tim)

1. Dilarang memakai: `transformers`, `nn.MultiheadAttention`, `nn.Transformer*`,
   `F.scaled_dot_product_attention`, bobot/embedding eksternal, API LLM.
2. Output insight di aplikasi **harus** berasal dari `MiniGPT.generate`.
   Lexicon hanya boleh dipakai untuk membuat data training dan untuk evaluasi.
3. Hanya satu GPU (`cuda:0`). Tidak ada multi-GPU atau distributed training.
4. Dataset asli read-only. Data hasil olahan dan checkpoint tidak di-commit.
5. Setiap modul punya unit test minimal sebelum dianggap selesai.
6. Fitur di luar §3 hanya dikerjakan jika MVP sudah berjalan end-to-end.

## 11. Rencana Implementasi Bertahap

Waktu tersisa ±2 hari. Prinsip: **jalur end-to-end secepatnya, lalu perbaiki**.

| # | Fase | Waktu | Output | Kriteria selesai |
|---|---|---|---|---|
| F0 | Setup | Sab 7 Nov, siang | requirements, .gitignore, struktur folder; upload dataset ke Kaggle | commit awal di GitHub; Kaggle Dataset siap |
| F1 | Data & silver label | Sab siang–sore | `preprocess.py`, `labeling.py`, `prepare_data.py` | train/val/test jsonl terbentuk; cek manual 30 sampel, ≥ ~80% label masuk akal |
| F2 | Tokenizer & dataset | Sab sore | `tokenizer.py`, `dataset.py` | test round-trip dan batas panjang lulus |
| F3 | Model | Sab sore–malam | `attention.py`, `model.py` | test causal mask lulus; overfit 1 batch (loss → ~0) di CPU |
| F4 | Training | Sab malam | `train.py`, `notebooks/train_kaggle.ipynb` | training di Kaggle GPU; test resume lulus; `best.pt` tersimpan sebagai output dan diunduh |
| F5 | Inferensi & evaluasi | Min 8 Nov, pagi | `inference.py`, `evaluate.py` | format_validity ≥ 0,90; `eval.json` terbentuk |
| F6 | Aplikasi | Min siang | `app/streamlit_app.py` | demo lokal berjalan dari `best.pt` (input teks & CSV) |
| F7 | Finalisasi | Min sore–malam | README, slide, contoh demo, rekaman cadangan | kode dibekukan; demo dicoba ulang dari clone bersih |
| — | Buffer & presentasi | Sen 9 Nov | — | — |

Pembagian kerja paralel (jika 3 orang):
- (A) F1 data dan label.
- (B) F2–F3 model.
- (C) F6 aplikasi dengan *stub* `InsightGenerator` sesuai INTERFACES.md, plus
  persiapan notebook Kaggle.

Setelah itu semua bergabung di F4–F5.

## 12. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Sesi Kaggle berakhir dan checkpoint hilang | `last.pt` disimpan tiap epoch; *Save Version* / unduh manual; resume otomatis (§9.2–9.3) |
| Kuota/antrian GPU Kaggle habis | Colab Free sebagai cadangan; model cukup kecil untuk dilatih di CPU dengan epoch dikurangi |
| Silver label berkualitas rendah | Validasi manual 30 sampel di F1; perbaiki lexicon sebelum training |
| Model mengeluarkan format rusak | Parser toleran, `max_new_tokens` terbatas, greedy decoding; laporkan format_validity |
| Data didominasi ulasan positif | Oversampling ulasan negatif pada split train |
| Waktu habis | F6 memakai UI paling sederhana; fitur tambahan dihapus terlebih dahulu |
| Dosen menganggap "rule-based" | Tegaskan bahwa aturan hanya untuk anotasi data; seluruh output aplikasi berasal dari generasi model; tunjukkan contoh generalisasi |

## 13. Keputusan yang Perlu Disetujui Tim

1. Pendekatan silver label dengan format sekuens terstruktur (§6).
2. Tokenizer word-level (bukan char-level/BPE).
3. Ukuran model (≈ 4,7–5,8 juta parameter) dan hyperparameter awal (§8).
4. Kaggle (GPU tunggal) untuk training utama, Colab sebagai cadangan, Streamlit lokal untuk demo (§9).
5. `Processed Dataset.csv` sebagai satu-satunya sumber data pipeline.
