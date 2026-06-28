import { pipeline } from '@huggingface/transformers';
import { TONE_CONFIG } from '../utils/config.js';
import { isWebGPUSupported, logError } from '../utils/common.js';

const MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct';

const TONE_PROMPTS = {
  normal: 'You are a helpful assistant. Share one interesting fun fact in 2-3 sentences.',
  funny: 'You are a comedian. Share one hilarious and funny fact with a joke twist in 2-3 sentences.',
  professional: 'You are a botanist scientist. Share one scientifically accurate fact in 2-3 sentences.',
  casual: 'You are a friendly person chatting casually. Share one cool fact in 2-3 sentences.',
};

export class RootFactsService {
  constructor() {
    this.generator = null;
    this.isModelLoaded = false;
    this.isGenerating = false;
    this.config = null;
    this.currentBackend = null;
    this.currentTone = TONE_CONFIG.defaultTone;
  }

  async loadModel() {
    const deviceOptions = [];
    if (isWebGPUSupported()) {
      deviceOptions.push('webgpu');
    }
    deviceOptions.push('wasm');

    let lastError;
    for (const device of deviceOptions) {
      try {
        this.generator = await pipeline('text-generation', MODEL_ID, {
          dtype: 'q4',
          device,
        });
        this.isModelLoaded = true;
        this.currentBackend = device;
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
    if (!this.isReady() || this.isGenerating) return null;

    this.isGenerating = true;

    try {
      const systemPrompt = TONE_PROMPTS[this.currentTone] || TONE_PROMPTS.normal;
      const messages = [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: `Tell me a fun fact about ${vegetableName}.` },
      ];

      const output = await this.generator(messages, {
        max_new_tokens: 150,
        temperature: 0.8,
        top_p: 0.9,
        do_sample: true,
        return_full_text: false,
      });

      const generatedText = output[0]?.generated_text;
      let factText;
      if (Array.isArray(generatedText)) {
        factText = generatedText.at(-1)?.content?.trim() || '';
      } else {
        factText = String(generatedText || '').trim();
      }

      return factText || null;
    } catch (error) {
      logError('RootFactsService.generateFacts', error);
      return null;
    } finally {
      this.isGenerating = false;
    }
  }

  isReady() {
    return this.generator !== null && this.isModelLoaded;
  }
}
