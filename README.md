# Gravar, salvar e separar por classe os áudios de direções a partir de um PDF em inglês

Importações:
    
    import pdfplumber
    import os
    import pathlib
    import pandas as pd
    import re
    import numpy as np
    import time
    import torch
    from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor
    import sys
    import librosa
    import pyaudio
    import IPython
    import webrtcvad
    import pyaudio
    import wave
    import speech_recognition as sr
    import noisereduce as nr

Lista todos os dispositivos de áudio disponíveis no sistema e mostra os nomes dos dispositivos que suportam o formato de áudio (taxa de amostragem de 48000Hz e formato de áudio paInt16)
    
    p = pyaudio.PyAudio()
    
    for i in range(p.get_device_count()): # número total de dispositivos de áudio disponíveis #Total number of audio devices available
        devinfo = p.get_device_info_by_index(i)  # nome, índice, número máximo de canais de entrada e formatos de áudio suportados #name, index, maximum number of input channels and audio formats supported
        
        try:
            if p.is_format_supported(48000,  # taxa de amostragem #sample rate
                                     input_device=devinfo['index'],
                                     input_channels=devinfo['maxInputChannels'],
                                     input_format=pyaudio.paInt16):
                print(p.get_device_info_by_index(i).get('name'))
        except Exception as e:
            continue

![Captura de tela 2024-05-03 175514](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/8da77508-8a15-4f07-9e2a-217307a3b1f6)

    all_clean_text = {} #dicionário vazio

PDF: text_directions
Note: Each code should be tailored to your PDF

![Captura de tela 2024-05-03 185206](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/a727eac3-7b09-4256-8226-149a86915763)


    def clean_text(extracted_text):
        text = extracted_text.replace('-\n', '')
        text = text.replace('\n', ' ')
        text = text.replace(' -', '')
        text = text.replace('— ', '')
        text = text[:text.rfind(' text_directions text_directions_c')]
        text = text.strip()
        return text

Leitura do PDF

    textpdf = 'temp'
    
    with pdfplumber.open('text_directions.pdf') as pdf:
        text = pdf.pages[0]
        
       
        if textpdf not in all_clean_text:
            all_clean_text[textpdf] = ''
        
         #Pega somente texto normal (ignora texto em negrito) #normal text(ignores bold text)
        text_filtered = text.filter(lambda obj: obj["object_type"] == "char" and not "Bold" in obj["fontname"])
        text_cleaned = clean_text(text_filtered.extract_text())
        
        all_clean_text[textpdf] += f' {text_cleaned}'

Função para a lista de microfones

    def list_microphones(pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
    
        result = []
        for i in range(0, numdevices):
            if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('name')
                result += [[i, name]]


Função 

    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]
    
    
            return result

Fluxo de áudio

    vad = webrtcvad.Vad()
    vad.set_mode(1) # (1) VAD agressivo: mais sensível à detecção de fala #(1) Aggressive VAD: more sensitive to speech detection
    
    audio = pyaudio.PyAudio()
    
    FORMAT = pyaudio.paInt16 # formato do áudio como inteiro de 16bits #audio format as 16-bit integer
    CHANNELS = 1 # número de canais de áudio como 1 (mono) #number of audio channels as 1 (mono)
    RATE = 48000 # taxa de amostragem do áudio como 48000 Hz #sample rate 4800Hz
    FRAME_DURATION = 10 # duração do quadro em milissegundos #frame duration in milliseconds
    CHUNK = int(RATE * FRAME_DURATION / 1000) # número de amostras de áudio por quadro #number of audio samples per frame
    
    
    device_name = 'HyperX SoloCast: USB Audio (hw:2,0)' # nome do dispositivo de áudio a ser usado #name of the audio device to be used
    microphones = list_microphones(audio)
    selected_input_device_id = get_input_device_id(
        device_name, microphones)
    
    stream = audio.open(input_device_index=selected_input_device_id,
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK) # abre o fluxo de áudio #open the audio stream
    stream.stop_stream() #fecha o fluxo de áudio #close the audio stream

Classe Wav2vec

    class Wave2Vec2Inference:
        def __init__(self,model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
            self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            if use_lm_if_possible:            
                self.processor = AutoProcessor.from_pretrained(model_name)
            else:
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = AutoModelForCTC.from_pretrained(model_name)
            self.model.to(self.device)
            self.hotwords = hotwords
            self.use_lm_if_possible = use_lm_if_possible
    
        def buffer_to_text(self, audio_buffer):
            if len(audio_buffer) == 0:
                return ""
    
            inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)
    
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device),
                                    attention_mask=inputs.attention_mask.to(self.device)).logits            
    
            if hasattr(self.processor, 'decoder') and self.use_lm_if_possible:
                transcription = \
                    self.processor.decode(logits[0].cpu().numpy(),                                      
                                          hotwords=self.hotwords,
                                          #hotword_weight=self.hotword_weight,  
                                          output_word_offsets=True,                                      
                                       )                             
                confidence = transcription.lm_score / len(transcription.text.split(" "))
                transcription = transcription.text       
            else:
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                confidence = self.confidence_score(logits,predicted_ids)
    
            return transcription, confidence   
    
        def confidence_score(self, logits, predicted_ids):
            scores = torch.nn.functional.softmax(logits, dim=-1)                                                           
            pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
            mask = torch.logical_and(
                predicted_ids.not_equal(self.processor.tokenizer.word_delimiter_token_id), 
                predicted_ids.not_equal(self.processor.tokenizer.pad_token_id))
    
            character_scores = pred_scores.masked_select(mask)
            total_average = torch.sum(character_scores) / len(character_scores)
            return total_average
    
        def file_to_text(self, filename):
            audio_input, samplerate = sf.read(filename)
            assert samplerate == 16000
            return self.buffer_to_text(audio_input)


Uso do HuggingFace

    vec = Wave2Vec2Inference("facebook/wav2vec2-large-960h-lv60-self")

Função que configura comandos para a gravação dos áudios

    silence_time = 0.3 # tempo máximo de silêncio antes de 'fechar' #maximum silence time before close the audio stream
    def listen_speech(text, df, file_id, path_to_save_data='data'):
        print(text)
        while True:
            stream.start_stream()
            tic = time.time() # tempo de início da captura. #audio capture start time
            frames = []
            f = b'' # armazenar o áudio acumulado em bytes. #store the accumulated audio in bytes
            while True:
                frame = stream.read(CHUNK, exception_on_overflow=False)
                is_speech = vad.is_speech(frame, RATE)
                if is_speech:
                    frames.append(frame)
                    f += frame
                    tic = time.time()
                elif time.time() - tic < silence_time:
                    continue # se não há fala, mas o tempo de silêncio permitido ainda não foi atingido, o loop continua. #if there is no speech, but the allowed silence time has not been reached yet, the loop continues
                else:
                    if len(frames) > 1:
                        if RATE == 16000:
                            audio_frames = f
                            float64_buffer = np.frombuffer(audio_frames, dtype=np.int16) / 32767 # converte o áudio em um array numpy de 64bits e normaliza os valores entre -1 e 1 #convert the audio into a 64-bit numpy array and normalize the values between -1 and 1
    
                            reduced_noise = nr.reduce_noise(y=float64_buffer, rate=RATE) # redução de ruídos #reduce noise
    
                            output_model = vec.buffer_to_text(float64_buffer) # envia o áudio processado para o modelo de reconhecimento de fala (vec) e obtém o texto reconhecido #sends the processed audio to the speech recognition (vec) model and retrieves the recognized text
                           
                        else:
                            waveFile = wave.open(f'{path_to_save_data}/audio/temp/temp.wav', 'wb')
                            
                            waveFile.setnchannels(CHANNELS)
                            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                            waveFile.setframerate(RATE)
                            waveFile.writeframes(b''.join(frames))
                            duration_seconds = waveFile.getnframes() / waveFile.getframerate()
                
                            waveFile.close()
    
                            audio_file, _ = librosa.load(f'{path_to_save_data}/audio/temp/temp.wav', sr=16000)
                            output_model = vec.buffer_to_text(audio_file)
                          
                        if len(output_model[0]) < 2 or output_model[1] < 0.8: # tamanho do texto reconhecido é maior que 2 caracteres e se a confiança do modelo é maior que 0.8 #if the size of the recognized text is greater than 2 characters and the model confidence is greater than 0.8
                            frames = []
                            tic = time.time()
                            continue
                        print(f'Wav2Vec2 result: {output_model}')
                        
                        stream.stop_stream()
    
                        # Converte audio do microfone em formato WAV e salva em disco #Converts microphone audio to WAV format and saves it to disk
                        
                        waveFile = wave.open(f'{path_to_save_data}/audio/{file_id}.wav', 'wb')
                        
                        waveFile.setnchannels(CHANNELS)
                        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                        waveFile.setframerate(RATE)
                        waveFile.writeframes(b''.join(frames))
                        duration_seconds = waveFile.getnframes() / waveFile.getframerate()
                        print(f'Duration: {duration_seconds}')
                        
                        waveFile.close()
                        break
                    frames = []
                    f = b''
           
            result_input = input()
            
            # Falar a mesma frase, ignorar a última frase #Speak the same phrase, ignore the last sentence
            
            if result_input == 'd':
                continue
                
            # Ignora e pula para a próxima sem salvar áudio/anotação #"Skip and move to the next one without saving audio/annotation
            
            elif result_input == 'n':
                os.remove(f'{path_to_save_data}/audio/{file_id}.wav')
                IPython.display.clear_output()
                return False, df, result_input
            else: # áudio é salvo e uma linha é adicionada ao DataFrame #Audio is saved and a line is added to the DataFrame
                IPython.display.clear_output()
                df = pd.concat([df, pd.DataFrame([[f'{path_to_save_data}/audio/{file_id}.wav', text.strip()]], columns=list(df.columns))])
                return True, df, result_input

Função para iniciar o caminho do arquivo

    def get_df_annotation(path='data'):
        if os.path.exists(f'{path}/annotation.tsv'):
            return pd.read_csv(f'{path}/annotation.tsv', sep='\t')
        return pd.DataFrame(columns=['path', 'sentence'])

Função para próximo arquivo

    def get_next_file_id(df_annotation):
        if len(df_annotation) > 0:
            last_filename = df_annotation.iloc[-1][0]
            if last_filename.rfind('/') == -1:
                return int(last_filename[:last_filename.find('.')]) + 1
            return int(last_filename[last_filename.rfind('/')+1:last_filename.find('.')]) + 1
        return 0

Função para salvar em anotação

    def save_annotation(is_to_save, df_annotation, path_to_save_data):
        if is_to_save:
            df_annotation.to_csv(f'{path_to_save_data}/annotation.tsv', sep='\t', index=None)
            return True
        return False

Função para retormar a gravação de onde parou

    count_search_where_stopped_default = 1
    def check_is_where_stopped(text, count_search_where_stopped):
        if len(df_annotation) > 0:
            if text.strip() == df_annotation.iloc[-count_search_where_stopped].sentence:
                count_search_where_stopped -= 1
                if count_search_where_stopped == 0:
                    return True, count_search_where_stopped
            else:
                count_search_where_stopped = count_search_where_stopped_default
        
        return False, count_search_where_stopped

Gravação

    # True == start from beginning; False == Continue from where stopped
    is_where_stopped = True  
    
    path_to_save_data = 'data'
    pathlib.Path(path_to_save_data).mkdir(exist_ok=True, parents=True)
    pathlib.Path(path_to_save_data + '/audio').mkdir(exist_ok=True, parents=True)
    pathlib.Path(path_to_save_data + '/audio/temp').mkdir(exist_ok=True, parents=True)
    df_annotation = get_df_annotation(path_to_save_data)
    file_id = get_next_file_id(df_annotation)
    result_input = ''
    
    count_search_where_stopped = count_search_where_stopped_default
    for tale_name, tale_text in all_clean_text.items():
        for text in tale_text.split('.'):
            if len(text) > 0:
                is_subsplited = False
                if is_subsplited is False:
                    if is_where_stopped is False:
                        is_where_stopped, count_search_where_stopped = check_is_where_stopped(text + '.', count_search_where_stopped)
                        continue
                    
                    is_file_saved, df_annotation, result_input = listen_speech(text + '.', df_annotation, file_id, path_to_save_data)
                    file_id += save_annotation(is_file_saved, df_annotation, path_to_save_data)
                    
                if result_input == 'end':
                    sys.exit()

Contagem de arquivos de áudio

    path_data = 'data/audio'
    
    total_time = 0
    total_files = 0
    for filename in os.listdir(path_data):
        if filename.endswith('wav'):
            audio_file, _ = librosa.load(f'{path_data}/{filename}', sr=16000)
            total_time += librosa.get_duration(y=audio_file, sr=16000)
            total_files += 1
    print(f'Total audio files: {total_files}')
    print(f'Mean time of audio files (sec): {total_time/total_files}')
    print(f'Total time audio files (min): {total_time/60}')

exemplo com 5 gravações

![Captura de tela 2024-05-03 175545](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/b76cee92-0b48-4344-b75c-3c1c9ef413df)

    from sklearn.model_selection import train_test_split


Função para o caminho para salvar em anotação

    def get_df_annotation(path='data'):
        if os.path.exists(f'{path}/annotation.tsv'):
            return pd.read_csv(f'{path}/annotation.tsv', sep='\t')
        return None

Função para salvar em treinamento

    def save_dataset_splited(df_annotation, filename='train'):
        df_annotation.to_csv(f'{filename}.tsv', sep='\t', index=None)

variável

    df_annotation = get_df_annotation('data')

Treinamento de teste 

    train, test = train_test_split(df_annotation, test_size=0.4)
    test, validation = train_test_split(test, test_size=0.5)

Salvar o dataset

    save_dataset_splited(train, 'train')
    save_dataset_splited(test, 'test')
    save_dataset_splited(validation, 'validation')

  ![Captura de tela 2024-05-03 180219](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/ec49e0c1-43f0-48c5-a589-a6f3defeeaa0)
  
![Captura de tela 2024-05-03 185114](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/f4b630c5-2ac3-430e-b881-84b3b7effc49)



  ![Captura de tela 2024-05-03 180210](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/e7222b22-3366-4d30-9373-70a74fa8764a)

![Captura de tela 2024-05-03 185105](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/a933ebe8-5dc3-44c5-a540-b07424bf308d)


  ![Captura de tela 2024-05-03 180224](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/8683ea8a-cccf-45c0-bb48-84f9f76431b4)

![Captura de tela 2024-05-03 185125](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/bc3a3c1e-20ff-4565-b7b1-ca7033a4ac82)


Explorador de arquivos do dispositivo

![Captura de tela 2024-05-03 185847](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/c0cb847a-3709-4007-b2b1-667ab38287dc)

![Captura de tela 2024-05-03 185856](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/266fa72f-043b-4317-8e02-28a43a0bdec7)

![Captura de tela 2024-05-03 185907](https://github.com/virnaaguiaar/audio_com_wav2vec/assets/85238057/5174849d-c24e-4f29-982b-6e32e115da47)


