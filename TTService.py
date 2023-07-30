import os
import sys
import time
sys.path.append('vits')
os.environ["PYTORCH_JIT"] = "0"

import torch
import soundfile
from loguru import logger

import vits.utils as utils
import vits.commons as commons
from vits.text.symbols import symbols
from vits.models import SynthesizerTrn
from vits.text import text_to_sequence

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


class TTService():
    def __init__(self, cfg, model, char, speed):
        logger.info('Initializing TTS Service for %s...' % char)
        self.hps = utils.get_hparams_from_file(cfg)
        self.speed = speed
        if torch.cuda.is_available():
            self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                **self.hps.model).cuda()
        else:
            self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                **self.hps.model).cpu()
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(model, self.net_g, None)

    def read(self, text):
        text = text.replace('~', '！')
        stn_tst = get_text(text, self.hps)
        with torch.no_grad():
            if torch.cuda.is_available():
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            else:
                x_tst = stn_tst.cpu().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
            audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.2, length_scale=self.speed)[0][
                0, 0].data.cpu().float().numpy()
        return audio

    def read_save(self, text, filename, sr):
        stime = time.time()
        au = self.read(text)
        soundfile.write(filename, au, sr)
        logger.info('VITS Synth Done, time used %.2f' % (time.time() - stime))

if __name__ == '__main__':
    import traceback
    from SentimentEngine import SentimentEngine

    tmp_recv_file = 'tmp/server_received.wav'
    tmp_proc_file = 'tmp/server_processed.wav'

    ## hard coded character map
    char_name = {
        'paimon': ['models/paimon6k.json', 'models/paimon6k_390k.pth', 'character_paimon', 1],
    }

    # TTS 语音合成
    tts = TTService(*char_name["paimon"])

    # Sentiment Engine 情感分析
    sentiment = SentimentEngine.SentimentEngine('SentimentEngine/models/paimon_sentiment.onnx')
    try:
        while 1:
            words = input("输入语音:")
            if words in ("q", "exit", "退出", "关闭"):
                break
            tts.read_save(words, tmp_proc_file, tts.hps.data.sampling_rate)
            logger.debug(words, "生成完毕")
        # s.tts.read_save("这里传入文字", s.tmp_proc_file, s.tts.hps.data.sampling_rate)
    except Exception as e:
        looger.error(e.__str__())
        logger.error(traceback.format_exc())
        raise e