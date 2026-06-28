import { pipeline } from '@huggingface/transformers';
import { TONE_CONFIG } from '../utils/config.js';
import { isWebGPUSupported, logError } from '../utils/common.js';

const MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct';
const GENERATION_TIMEOUT_MS = 20000;

const VEGETABLE_NAMES_ID = {
  Beetroot: 'bit merah',
  Paprika: 'paprika',
  Cabbage: 'kubis',
  Carrot: 'wortel',
  Cauliflower: 'kembang kol',
  Chilli: 'cabai',
  Corn: 'jagung',
  Cucumber: 'mentimun',
  eggplant: 'terong',
  Garlic: 'bawang putih',
  Ginger: 'jahe',
  Lettuce: 'selada',
  Onion: 'bawang merah',
  Peas: 'kacang polong',
  Potato: 'kentang',
  Turnip: 'lobak',
  Soybean: 'kedelai',
  Spinach: 'bayam',
};

const FALLBACK_FACTS = {
  Beetroot: 'Bit merah mengandung betanin yang memberi warna merah tua dan berfungsi sebagai antioksidan kuat. Jus bit terbukti meningkatkan stamina atletik hingga 16% karena kandungan nitratnya yang memperlancar aliran darah ke otot.',
  Paprika: 'Paprika merah mengandung vitamin C tiga kali lebih banyak dari jeruk! Warna paprika ditentukan tingkat kematangannya — hijau paling muda, kuning menengah, dan merah paling matang dengan rasa paling manis.',
  Cabbage: 'Kubis telah dibudidayakan selama lebih dari 4.000 tahun dan dikenal sebagai "obat rakyat" di Eropa kuno. Kandungan vitamin K-nya sangat tinggi sehingga berperan penting dalam pembekuan darah dan kesehatan tulang.',
  Carrot: 'Wortel aslinya berwarna ungu atau kuning — wortel oranye baru dikembangkan di Belanda pada abad ke-17. Beta-karoten dalam wortel diubah tubuh menjadi vitamin A yang menjaga kesehatan mata dan sistem imun.',
  Cauliflower: 'Kembang kol adalah sayuran serbaguna yang bisa diolah menjadi pengganti nasi hingga pizza. Satu porsi kembang kol memenuhi 77% kebutuhan harian vitamin C dan kaya senyawa sulforafan yang bersifat antikanker.',
  Chilli: 'Cabai mengandung capsaicin yang memicu pelepasan endorfin — hormon kebahagiaan alami di otak. Ironisnya, capsaicin juga digunakan sebagai bahan pereda nyeri dalam krim topikal karena sifat antiinflamasinya.',
  Corn: 'Jagung adalah satu-satunya sereal yang dianggap sebagai sayuran, buah, dan biji-bijian sekaligus tergantung cara penggunaannya. Setiap tongkol jagung selalu memiliki jumlah baris yang genap, biasanya 16 baris.',
  Cucumber: 'Mentimun terdiri dari 96% air, menjadikannya salah satu makanan paling menghidrasi. Suhu di dalam mentimun bisa 20 derajat lebih sejuk dari udara luar — itulah asal ungkapan "cool as a cucumber".',
  eggplant: 'Terong secara botanis adalah buah beri, bukan sayuran! Kulit terong yang berwarna ungu mengandung nasunin, antioksidan yang terbukti melindungi sel otak dari kerusakan radikal bebas.',
  Garlic: 'Bawang putih telah digunakan sebagai obat selama lebih dari 5.000 tahun di Mesir, Yunani, dan China. Senyawa allicin yang terbentuk saat bawang putih dihancurkan adalah antibakteri alami yang ampuh melawan banyak jenis bakteri.',
  Ginger: 'Jahe telah digunakan dalam pengobatan tradisional selama lebih dari 2.000 tahun untuk mengatasi mual dan peradangan. Kandungan gingerol dalam jahe segar memiliki efek antiinflamasi yang lebih kuat dari beberapa obat pereda nyeri ringan.',
  Lettuce: 'Selada adalah salah satu sayuran tertua yang dibudidayakan manusia, dengan catatan sejarah sejak 2.700 SM di Mesir kuno. Kandungan laktoserin dalam selada memberikan efek menenangkan ringan yang dapat membantu tidur lebih nyenyak.',
  Onion: 'Bawang merah telah digunakan sebagai makanan dan obat sejak 5.000 tahun lalu. Senyawa quercetin dalam bawang merah adalah antioksidan kuat yang membantu menurunkan tekanan darah dan melindungi kesehatan jantung.',
  Peas: 'Kacang polong adalah salah satu tanaman pertama yang dibudidayakan manusia sekitar 10.000 tahun lalu. Penelitian Mendel tentang kacang polong menjadi fondasi ilmu genetika modern yang kita kenal hingga sekarang.',
  Potato: 'Kentang berasal dari Pegunungan Andes Amerika Selatan dan telah dibudidayakan selama 8.000 tahun. Kulit kentang mengandung lebih banyak nutrisi dari dagingnya, termasuk serat, kalium, dan vitamin C.',
  Turnip: 'Lobak adalah sayuran akar yang tahan banting dan bisa tumbuh di tanah miskin nutrisi sekalipun. Di Eropa, lobak adalah makanan pokok sebelum kentang populer dan bahkan pernah digunakan sebagai lentera pada perayaan Halloween.',
  Soybean: 'Kedelai adalah satu-satunya tanaman nabati yang mengandung semua 9 asam amino esensial, menjadikannya protein lengkap setara daging. Satu cangkir kedelai mengandung lebih banyak protein dari kebanyakan jenis daging merah.',
  Spinach: 'Bayam kaya akan vitamin K, zat besi, dan antioksidan yang mendukung kesehatan tulang dan darah. Satu mangkuk bayam memenuhi lebih dari 1.000% kebutuhan vitamin K harian yang penting untuk proses pembekuan darah.',
};

function buildMessages(vegetableName, nameId, tone) {
  const toneStyle = {
    normal: 'informatif dan menarik',
    funny: 'lucu dan menghibur dengan sentuhan humor ringan',
    professional: 'ilmiah dan akurat seperti seorang ahli botani',
    casual: 'santai dan ramah seperti teman yang berbagi cerita',
  }[tone] || 'informatif dan menarik';

  return [
    {
      role: 'system',
      content: `Kamu adalah asisten yang memberikan fakta menarik tentang sayuran dalam bahasa Indonesia. Selalu berikan fakta yang spesifik dan relevan hanya tentang sayuran yang diminta. Jangan membahas sayuran lain.`,
    },
    {
      role: 'user',
      content: `Berikan satu fakta menarik tentang ${vegetableName} (nama Indonesia: ${nameId}) dalam bahasa Indonesia. Gaya: ${toneStyle}. Fakta harus spesifik tentang ${vegetableName} dan terdiri dari 2-3 kalimat. Mulai langsung dengan faktanya tanpa awalan seperti "Tentu" atau "Berikut".`,
    },
  ];
}

function buildRetryMessages(vegetableName, nameId) {
  return [
    {
      role: 'system',
      content: `Jawab HANYA tentang ${vegetableName} (${nameId}). Dilarang membahas sayuran lain.`,
    },
    {
      role: 'user',
      content: `Fakta menarik tentang ${vegetableName} dalam 2 kalimat bahasa Indonesia:`,
    },
  ];
}

function isRelevant(text, vegetableName, nameId) {
  if (!text || text.length < 30) return false;
  const lower = text.toLowerCase();
  return (
    lower.includes(vegetableName.toLowerCase()) ||
    lower.includes(nameId.toLowerCase())
  );
}

export class RootFactsService {
  constructor() {
    this.generator = null;
    this.isModelLoaded = false;
    this.isGenerating = false;
    this.currentTone = TONE_CONFIG.defaultTone;
  }

  async loadModel() {
    const deviceOptions = isWebGPUSupported() ? ['webgpu', 'wasm'] : ['wasm'];

    let lastError;
    for (const device of deviceOptions) {
      try {
        this.generator = await pipeline('text-generation', MODEL_ID, {
          dtype: 'q4',
          device,
        });
        this.isModelLoaded = true;
        return true;
      } catch (error) {
        lastError = error;
        logError(`RootFactsService.loadModel (${device})`, error);
      }
    }

    throw lastError;
  }

  setTone(tone) {
    this.currentTone = tone;
  }

  async generateFacts(vegetableName) {
    const nameId = VEGETABLE_NAMES_ID[vegetableName] || vegetableName;
    const fallback = FALLBACK_FACTS[vegetableName];

    // If LLM not ready, immediately return pre-defined fact
    if (!this.isReady() || this.isGenerating) {
      return fallback || null;
    }

    this.isGenerating = true;

    try {
      // First attempt
      let result = await this._generateWithTimeout(
        buildMessages(vegetableName, nameId, this.currentTone),
      );

      if (isRelevant(result, vegetableName, nameId)) {
        return result;
      }

      // Retry with stricter prompt
      result = await this._generateWithTimeout(
        buildRetryMessages(vegetableName, nameId),
      );

      if (isRelevant(result, vegetableName, nameId)) {
        return result;
      }

      // LLM output not relevant — use pre-defined fallback
      return fallback || result;
    } catch (error) {
      logError('RootFactsService.generateFacts', error);
      return fallback || null;
    } finally {
      this.isGenerating = false;
    }
  }

  async _generateWithTimeout(messages) {
    const timeout = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('generation timeout')), GENERATION_TIMEOUT_MS),
    );

    const generation = this.generator(messages, {
      max_new_tokens: 150,
      temperature: 0.7,
      top_p: 0.9,
      do_sample: true,
      return_full_text: false,
    });

    const output = await Promise.race([generation, timeout]);
    const generated = output[0]?.generated_text;

    if (Array.isArray(generated)) {
      return generated.at(-1)?.content?.trim() || '';
    }
    return String(generated || '').trim();
  }

  isReady() {
    return this.generator !== null && this.isModelLoaded;
  }
}
