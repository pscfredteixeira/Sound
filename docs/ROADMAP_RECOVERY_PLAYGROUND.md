# SampleForge AI — roadmap de recuperação e criação musical

## Objetivo adicional: recuperar samples de músicas antigas

Uma faixa finalizada não permite recuperar exatamente todos os samples originais. A separação de stems é uma **estimativa** e a reconstrução de timbres só pode produzir aproximações. O objetivo é maximizar material musical utilizável, sem inventar sons ou categorias.

### Fluxo proposto: Sample Recovery

1. Importar uma faixa produzida pelo usuário (WAV/FLAC preferencialmente; MP3 quando necessário).
2. Separar Drums, Bass, Vocals e Other com um modelo comprovado de quatro stems.
3. Detectar transientes e trechos estáveis. Identificar eventos potencialmente isolados e regiões com interferência.
4. Propor one-shots, loops e frases; conservar original e versão processada quando houver limpeza.
5. Dar nota de confiança **relativa** (não probabilidade calibrada) e permitir escuta A/B.
6. Quando a reconstrução for possível, oferecer análise de pitch/envelope/ambiente e ressíntese **aproximada**, claramente rotulada.
7. Exportar WAV, metadados JSON e stems originais gerados pelo modelo.

A ausência do sample original impede comparação exata de timbre e não justifica promessas de fidelidade perfeita.

## Sample Playground — sequenciador para brincar com os samples

**Fase proposta: 3B, após validar o slicer da etapa 2.** Não faz parte do MVP 0.1. Nenhum botão de demonstração deve fingir reproduzir áudio.

### Dois modos

**Step Sequencer:** grade de 16/32/64 passos; cada pista recebe um sample real, com mute/solo, volume e panorama, padronização de BPM, swing e velocity por passo. Resolução de 1/16 no compasso 4/4 com quantização configurável.

**Timeline multifaixa:** arrastar samples para pistas; mover, duplicar, silenciar, cortar, ajustar fades e criar sequências por compassos. Snap to grid e zoom.

### Motor de áudio (necessário para implementação real)

- Engine de reprodução baseada em amostras, com relógio em frames de áudio e *não* em timers gráficos. Uma opção é PortAudio por `sounddevice` (após verificar suporte WASAPI no Windows).
- Pré-decodificar e converter samples para uma frequência interna escolhida (ex.: 48 kHz), com buffers limitados para economizar memória.
- Agendar eventos de uma grade a partir da fórmula `frames_per_16th = sample_rate * 60 / (BPM * 4)`; usar acumulador fracionário para evitar deriva de tempo.
- Mixer de pistas com ganho e limites para evitar clipping; crossfade só onde necessário, preservando transientes.
- Off-line render para WAV, evitando diferenças entre preview e exportação; salvar projeto editável em JSON.

### Critérios de aceite

- Uma sequência de kick/hat/bass gerada **a partir de arquivos reais** toca sem interrupções audíveis.
- BPM e posições dos eventos são estáveis, com erro de agendamento menor do que um bloco de áudio.
- Exportação multifaixa ou mixdown WAV pode ser aberta no Ableton Live 12.
- Salvar e reabrir um projeto mantém o arranjo.
- Ausência de samples ou drivers produz erro explícito, não playback falso.

## Critérios do MVP 0.1

- Interface abre em Windows 11, importa e toca WAV, MP3 e FLAC compatíveis com o backend de mídia.
- No primeiro uso, os pesos são baixados para cache local; modelos já baixados funcionam sem internet.
- Uma música autorizada gera quatro **WAVs reais** (Drums, Bass, Vocals, Other).
- Cada stem pode ser ouvido; exportação 44.1/48 kHz 16/24-bit cria JSON.
- Cancelamento e ausência de FFmpeg mostram mensagens úteis.
- Teste de ponta a ponta em máquina Windows real continua necessário, incluindo qualidade e memória de GPU.

## Evolução

- **Etapa 1:** núcleo desktop de separação e exportação (código implementado, teste real de inferência em Windows pendente).
- **Etapa 2:** recortes, transientes, BPM/beat grid, editor e preview de samples.
- **Etapa 3:** classificação com confiança, deduplicação, recuperação assistida e Auto Pack Generator.
- **Etapa 3B:** Sample Playground — sequenciador e timeline com playback real.
- **Etapa 4:** instalador Windows, diagnósticos, regressões e otimização.

Privacidade: tudo roda localmente; a primeira aquisição de modelos pode acessar os servidores de download. Os arquivos originais nunca são sobrescritos.
