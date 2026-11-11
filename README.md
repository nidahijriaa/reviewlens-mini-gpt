# ReviewLens GPT

**Pembuat insight dari ulasan pelanggan berbahasa Indonesia, memakai Mini-GPT yang dilatih dari nol.**

Tugas mata kuliah Generative AI · Tim 2 orang · Tenggat 11 November 2026

---

## 1. Tentang Proyek

### Masalahnya
Toko online menerima ribuan ulasan pelanggan. Membaca semuanya satu per satu untuk
mengetahui apa yang disukai dan dikeluhkan pembeli memakan waktu lama.

### Idenya
Kami melatih sebuah model bahasa kecil (Mini-GPT) yang "membaca" sebuah ulasan, lalu
**menuliskan sendiri** ringkasannya dalam tiga jenis insight:

| Insight | Pertanyaan yang dijawab | Contoh keluaran |
|---|---|---|
| **Positive Insights** | Apa yang disukai pelanggan? | "pengiriman cepat", "barang sesuai deskripsi" |
| **Customer Pain Points** | Apa yang dikeluhkan? | "produk rusak atau cacat", "pengiriman lambat" |
| **Improvement Suggestions** | Apa yang sebaiknya diperbaiki? | "perketat quality control sebelum pengiriman" |

Insight dari banyak ulasan kemudian dihitung frekuensinya menjadi sebuah
**Customer Insight Report**: daftar insight yang paling sering muncul beserta
persentase ulasan yang menyebutnya.

---

## 2. Persyaratan Teknis yang Dipenuhi

| Persyaratan | Cara dipenuhi | Lokasi |
|---|---|---|
| Decoder-only Transformer | `MiniGPT`: token embedding + positional embedding → 4 blok Transformer (pre-LayerNorm) → LayerNorm → LM head | [src/reviewlens/model.py](src/reviewlens/model.py) |
| Multi-head self-attention ditulis manual | Q, K, V dan `softmax(QKᵀ/√d)·V` dihitung sendiri; **tidak** memakai `nn.MultiheadAttention`, `nn.Transformer*`, atau `F.scaled_dot_product_attention` | [src/reviewlens/attention.py](src/reviewlens/attention.py) |
| Causal mask | Matriks segitiga bawah; posisi "masa depan" diisi `-inf` sebelum softmax, sehingga model tidak bisa mengintip kata berikutnya. Dibuktikan dengan unit test | [attention.py](src/reviewlens/attention.py), [tests/test_attention.py](tests/test_attention.py) |
| Next-token prediction | Model belajar menebak token berikutnya: `x = ids[:-1]`, `y = ids[1:]`, loss cross-entropy (padding target diabaikan dengan `-100`) | [model.py](src/reviewlens/model.py), [dataset.py](src/reviewlens/dataset.py) |
| Bobot acak dari nol | Semua bobot diinisialisasi acak Normal(0; 0,02); tidak memuat bobot dari luar | [model.py](src/reviewlens/model.py) |
| Tanpa pretrained model | Tokenizer dan embedding dibangun dari data ulasan sendiri; tidak memakai `transformers` atau model siap pakai | [src/reviewlens/tokenizer.py](src/reviewlens/tokenizer.py) |
| Tanpa API LLM | Semua insight dihasilkan oleh `MiniGPT.generate` dari checkpoint sendiri, sepenuhnya lokal | [notebooks/02_train_evaluate.ipynb](notebooks/02_train_evaluate.ipynb) |

Konfigurasi model bawaan ([config.py](src/reviewlens/config.py)): 4 lapisan, 4 head,
dimensi 256, feed-forward 1024, panjang konteks 128 token, dropout 0,1.

---

## 3. Alur Sistem (Bahasa Sederhana)

```
 Data ulasan ──► Silver label ──► Tokenizer ──► MiniGPT ──► Generate ──► Customer Insight Report
```

1. **Data.** Kami memakai `Processed Dataset.csv` berisi ulasan produk elektronik
   (Lazada Indonesia) beserta label sentimennya. Teks dibersihkan: huruf kecil, slang
   dinormalkan (misalnya `gak` → `tidak`), dan ulasan kosong atau duplikat dibuang.
   Ulasan yang teksnya sama tetapi label sentimennya bertentangan dibuang, dan ulasan
   dengan teks sama selalu ditaruh di split yang sama agar tidak "bocor" ke data uji.
2. **Silver label.** Dataset tidak punya contoh insight. Karena itu kami membuatnya
   otomatis dengan aturan kata kunci untuk 9 aspek (kualitas, kerusakan, pengiriman,
   kemasan, kesesuaian deskripsi, harga, pelayanan penjual, keaslian, garansi).
   Label otomatis ini disebut *silver label*: cukup baik untuk belajar, tetapi bukan
   anotasi manusia. **Aturan ini hanya dipakai untuk membuat data latih dan untuk evaluasi.**
3. **Tokenizer.** Setiap kata diubah menjadi nomor (ID). Kamus kata (vocabulary) dibangun
   hanya dari data latih. Ulasan dan targetnya disusun menjadi satu urutan:
   ```
   <bos> <review> barang bagus tapi pengiriman lama sekali
   <pos> kualitas produk bagus <neg> pengiriman lambat <sug> percepat proses pengiriman <eos>
   ```
4. **MiniGPT.** Model dilatih dari nol untuk menebak token berikutnya pada urutan di atas.
   Dengan cara itu ia belajar memetakan ulasan menjadi insight.
5. **Generate.** Untuk ulasan baru, model diberi awalan `<bos> <review> … <pos>` lalu
   menulis sisanya sendiri, satu token demi satu token. Hasilnya dipecah menjadi tiga bagian insight.
6. **Customer Insight Report.** Insight dari banyak ulasan dihitung frekuensinya dan
   ditampilkan sebagai top 5 per kategori beserta persentasenya.

---

## 4. Struktur Folder

```
reviewlens-mini-gpt/
├── README.md                  # dokumen ini
├── requirements.txt           # dependency Python
├── PROJECT_CONTEXT.md         # rencana proyek, keputusan desain, status akhir (§14)
├── INTERFACES.md              # kontrak antar-modul (signature fungsi, format data)
├── src/reviewlens/
│   ├── config.py              # konfigurasi model & training, lokasi file, pemilihan device, seed, IGNORE_INDEX
│   ├── preprocess.py          # baca CSV, pembersihan teks, normalisasi slang/negasi, grouped split
│   ├── labeling.py            # silver label: lexicon 9 aspek → positives / pain points / suggestions
│   ├── tokenizer.py           # tokenizer word-level + 10 token khusus; CLI untuk membangun vocab.json
│   ├── dataset.py             # format urutan, pemotongan panjang, Dataset & collate_fn (padding)
│   ├── attention.py           # multi-head self-attention manual + causal mask
│   └── model.py               # FeedForward, TransformerBlock, MiniGPT (forward, loss, generate)
├── scripts/
│   └── prepare_data.py        # CSV → train.jsonl / val.jsonl / test.jsonl (+ statistik split)
├── tests/                     # unit test semua modul di src/ dan scripts/ (data sintetis)
└── notebooks/
    ├── train_kaggle.ipynb               # pemeriksaan awal Kaggle: GPU tersedia & kolom dataset lengkap
    ├── 01_prepare_data_validation.ipynb # menjalankan prepare_data.py pada data asli & memvalidasi split
    ├── 02_train_evaluate.ipynb          # vocab, training 3 eksperimen, evaluasi, demo Customer Insight Report
    └── 03_epoch_batch_experiment.ipynb  # eksperimen lanjutan: 20 epoch (E4) dan batch 128 (E5)
```

Dataset asli, data olahan (`data/`), dan checkpoint (`*.pt`) **tidak** disimpan di GitHub
(lihat [.gitignore](.gitignore)).

---

## 5. Cara Menjalankan Notebook di Kaggle

Setiap notebook mengambil source code terbaru dengan `git clone` dari GitHub (branch `main`),
jadi tidak perlu mengunggah kode secara manual.

1. Unggah `Processed Dataset.csv` sebagai **Kaggle Dataset privat** (cukup sekali).
2. Buka notebook di Kaggle (*File → Import Notebook*, lalu pilih file `.ipynb` dari folder `notebooks/`).
3. Di panel kanan:
   - **Add Input** → pilih dataset yang berisi `Processed Dataset.csv` → *Add*.
     Notebook akan mencari file itu sendiri di `/kaggle/input/`.
   - **Settings → Accelerator → GPU T4** (cukup satu GPU; proyek ini hanya memakai `cuda:0`).
   - **Settings → Internet → ON** (dibutuhkan untuk `git clone`).
4. Klik **Save Version → Save & Run All (Commit)**. Notebook berjalan sampai selesai di
   latar belakang, dan semua file di `/kaggle/working` tersimpan sebagai *output* versi itu.
   Tanpa *Save Version*, isi `/kaggle/working` hilang saat sesi berakhir.

Urutan yang disarankan: `train_kaggle` (cek lingkungan) → `01` (validasi data) → `02`
(training & evaluasi) → `03` (eksperimen lanjutan).

Hasil utama notebook 02 ada di `/kaggle/working/runs/`:
`runs/<nama_eksperimen>/best.pt`, `last.pt`, `history.json`, `vocab.json`, serta
`runs/results.json` dan `runs/eval.json`.

Menjalankan persiapan data saja (tanpa notebook):
```bash
python scripts/prepare_data.py --csv "/kaggle/input/<slug>/Processed Dataset.csv" --out-dir /kaggle/working/data/processed
python -m reviewlens.tokenizer --data-dir /kaggle/working/data/processed   # dengan src/ di PYTHONPATH
```

---

## 6. Menjalankan Unit Test secara Lokal

Unit test hanya memakai data sintetis; dataset asli tidak diperlukan dan GPU tidak wajib.

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pytest -q tests
```

Untuk komputer tanpa GPU, PyTorch versi CPU (lebih kecil) dapat dipasang lebih dulu:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

---

## 7. Ringkasan Hasil

Angka di bawah berasal dari output notebook 01 dan 02 yang dijalankan di Kaggle (GPU T4).

### 7.1 Data (notebook 01)

| Keterangan | Jumlah |
|---|---|
| Ulasan mentah | 41.711 |
| Ulasan yang dipakai | 39.210 |
| Kelompok konflik (teks sama, sentimen bertentangan) yang dibuang | 118 kelompok (1.845 baris) |
| Split train / validation / test | 31.368 / 3.921 / 3.921 |
| Teks identik yang muncul di dua split berbeda | 0 |
| Ukuran vocabulary | 5.970 token |

### 7.2 Eksperimen training (notebook 02; 6 epoch, batch 64)

| Eksperimen | Perbedaan | Best validation loss |
|---|---|---|
| E1_baseline | konfigurasi bawaan, learning rate 3e-4 | 2,4309 |
| **E2_lr_tinggi** | learning rate 1e-3 | **2,3741** (terbaik) |
| E3_model_kecil | 2 lapisan, dimensi 128, feed-forward 512 | 2,5633 |

Model terbaik dipilih berdasarkan validation loss (makin kecil makin baik).

### 7.3 Evaluasi model terbaik (E2) pada data test

| Metrik | Arti sederhana | Nilai |
|---|---|---|
| `test_perplexity` | Seberapa "bingung" model menebak token berikutnya (makin kecil makin baik) | 10,991 |
| `format_validity` | Proporsi output yang formatnya benar (`<pos> … <neg> … <sug> … <eos>`) | 1,000 |
| `aspect_f1` positives | Kecocokan frasa positif dengan silver label (0–1) | 0,927 |
| `aspect_f1` pain_points | Kecocokan frasa keluhan dengan silver label | 0,772 |
| `aspect_f1` suggestions | Kecocokan frasa saran dengan silver label | 0,774 |
| `aspect_f1` micro | Gabungan ketiganya | 0,875 |
| `exact_match` | Proporsi ulasan yang ketiga insight-nya persis sama dengan silver label | 0,800 |

Metrik berbasis generate dihitung dengan greedy decoding. Pembandingnya adalah silver label,
bukan anotasi manusia (lihat Keterbatasan).

### 7.4 Notebook 03 (eksperimen lanjutan)

Notebook 03 menguji apakah training lebih lama (**E4: 20 epoch**) atau batch lebih besar
(**E5: batch size 128**) memperbaiki hasil dibandingkan eksperimen 6 epoch / batch 64 di atas.
Hasilnya belum tersedia saat README ini ditulis.

---

## 8. Perubahan dari Rencana Awal

| Rencana awal | Yang dilakukan | Alasan |
|---|---|---|
| Modul `train.py`, `inference.py`, `evaluate.py` di `src/reviewlens/` | Fungsi training, inferensi (generate + parsing), dan evaluasi ditulis di dalam `notebooks/02_train_evaluate.ipynb` | Training dijalankan sebagai notebook di GPU Kaggle |
| Aplikasi Streamlit (`app/streamlit_app.py`) | Tidak dibuat; demo memakai Customer Insight Report di notebook 02 | Keterbatasan waktu |
| Checkpoint di folder `checkpoints/` | Checkpoint disimpan per eksperimen di `runs/<nama_eksperimen>/` | Notebook menjalankan beberapa eksperimen sekaligus, sehingga tiap eksperimen perlu folder sendiri |
| `tests/test_checkpoint.py` dan `tests/test_inference.py` | Tidak dibuat; pemeriksaan checkpoint dan parsing dilakukan di dalam notebook 02 | Kode checkpoint dan inferensi berada di notebook, bukan di modul yang bisa di-test |
| Tenggat 9 November 2026 | Tenggat diperpanjang menjadi 11 November 2026; proyek dikerjakan tim 2 orang | Perpanjangan tenggat |

---

## 9. Keterbatasan

- **Evaluasi memakai silver label, bukan anotasi manusia.** Skor F1 dan exact match
  mengukur seberapa mirip output model dengan label otomatis berbasis aturan. Jika
  aturannya keliru, kesalahan itu juga ada di data latih dan di kunci jawaban.
- **Dataset hanya berisi ulasan produk elektronik.** Model belum tentu bekerja baik untuk
  kategori lain (fesyen, makanan, dan sebagainya) yang memakai kosakata berbeda.
- **Tokenizer word-level.** Kata yang tidak ada di vocabulary (misalnya typo baru atau
  istilah langka) diubah menjadi `<unk>`, sehingga informasinya hilang bagi model.
