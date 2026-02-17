class AudioProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const pcm = input[0];
      const buffer = new Int16Array(pcm.length);
      for (let i = 0; i < pcm.length; i++) {
        let s = Math.max(-1, Math.min(1, pcm[i]));
        buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      this.port.postMessage(buffer.buffer, [buffer.buffer]);
    }
    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);
