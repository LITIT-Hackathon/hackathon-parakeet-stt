# Test fixture attribution

- **`tone_16k.wav`** — a synthetic tone. Drives the stub tier and the
  audio-layer tests (resample / downmix / validation); no speech content.

- **`speech_en.wav`** — a public-domain reading (from parakeet.cpp's own test
  fixtures; text is Hawthorne, *The House of the Seven Gables*). Exercises the
  native English path; the word "portrait" is asserted.

- **`speech_de.wav`** — an approximately 12-second excerpt of Deutsche Welle
  *Langsam gesprochene Nachrichten* (learngerman.dw.com), the German test audio
  the challenge points to. © Deutsche Welle; included as a short excerpt for
  automated testing only. Exercises the native German path; the word
  "Sonnenfinsternis" is asserted.
