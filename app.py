from flask_api import status
from flask import Flask,jsonify
from flask import request, redirect, url_for
from flask_cors import CORS
import logging
logging.basicConfig(level= logging.INFO)
from modules.Hi_MAP import translate
from nltk.tokenize import word_tokenize

import argparse
import sys
import os
from translate_infer import build_translator
import re

app = Flask(__name__)
CORS(app)

def clean_summary_str(s):
    s = s.lower()
    s = s.replace('<unk>','')
    s = s.replace('<blank>','')
    s = s.replace('`', '')
    s = s.replace('.', '')
    s = s.replace(',', '')
    s = s.replace(';', '')
    s = s.replace('\'', '')
    s = s.replace('\"', '')
    s = s.replace('(', '')
    s = s.replace(')', '')
    s = s.replace('-', ' ')
    s = s.replace('<p>', '')
    s = s.replace('</p>', '')
    s = s.replace('<t>', '')
    s = s.replace('</t>', '')
    s = s.replace('[!@#$]', '')
    return s.rstrip()

def preprocess(s):
    s= s.lower()
    w_tokens = word_tokenize(s)
    return " ".join(text for text in w_tokens)
    
class DeprecateAction(argparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.mdhelp is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)

def translate_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add_argument('-model', dest='models', metavar='MODEL',
                       nargs='+', type=str, default=["export_models/newser_mmr/Feb17__step_20000.pt"], 
                       help='Path to model .pt file(s). '
                            'Multiple models can be specified, '
                            'for ensemble decoding.')

    group = parser.add_argument_group('Data')
    group.add_argument('-data_type', default="text",
                       help="Type of the source input. Options: [text|img].")

    group.add_argument('-src', default ="preprocessed_truncated/test.txt.src.tokenized.fixed.cleaned.final.truncated.txt",
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add_argument('-src_dir', default="",
                       help='Source directory for image or audio files')
    group.add_argument('-tgt',
                       help='True target sequence (optional)')
    group.add_argument('-output', default='pred.txt',
                       help="""Path to output the predictions (each line will
                       be the decoded sequence""")
    group.add_argument('-report_bleu', action='store_true',
                       help="""Report bleu score after translation,
                       call tools/multi-bleu.perl on command line""")
    group.add_argument('-report_rouge', action='store_true',
                       help="""Report rouge 1/2/3/L/SU4 score after translation
                       call tools/test_rouge.py on command line""")

    # Options most relevant to summarization.
    group.add_argument('-dynamic_dict', action='store_true',
                       help="Create dynamic dictionaries")
    group.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    group = parser.add_argument_group('Beam')
    group.add_argument('-fast', action="store_true",
                       help="""Use fast beam search (some features may not be
                       supported!)""")
    group.add_argument('-beam_size', type=int, default=4,
                       help='Beam size')
    group.add_argument('-min_length', type=int, default=200,
                       help='Minimum prediction length')
    group.add_argument('-max_length', type=int, default=300,
                       help='Maximum prediction length.')
    group.add_argument('-max_sent_length', action=DeprecateAction,
                       help="Deprecated, use `-max_length` instead")

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add_argument('-stepwise_penalty', action='store_true',
                       help="""Apply penalty at every decoding step.
                       Helpful for summary penalty.""")
    group.add_argument('-length_penalty', default='wu',
                       choices=['none', 'wu', 'avg'],
                       help="""Length Penalty to use.""")
    group.add_argument('-coverage_penalty', default='summary',
                       choices=['none', 'wu', 'summary'],
                       help="""Coverage Penalty to use.""")
    group.add_argument('-alpha', type=float, default=0.9,
                       help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
    group.add_argument('-beta', type=float, default=5,
                       help="""Coverage penalty parameter""")
    group.add_argument('-block_ngram_repeat', type=int, default=3,
                       help='Block repetition of ngrams during decoding.')
    group.add_argument('-ignore_when_blocking', nargs='+', type=str,
                       default=['story_separator_special_tag'],
                       help="""Ignore these strings when blocking repeats.
                       You want to block sentence delimiters.""")
    group.add_argument('-replace_unk', action="store_true",
                       help="""Replace the generated UNK tokens with the
                       source token that had highest attention weight. If
                       phrase_table is provided, it will lookup the
                       identified source token and give the corresponding
                       target token. If it is not provided(or the identified
                       source token does not exist in the table) then it
                       will copy the source token""")

    group = parser.add_argument_group('Logging')
    group.add_argument('-verbose', action="store_true",
                       help='Print scores and predictions for each sentence')
    group.add_argument('-log_file', type=str, default="",
                       help="Output logs to a file under this path.")
    group.add_argument('-attn_debug', action="store_true",
                       help='Print best attn for each word')
    group.add_argument('-dump_beam', type=str, default="",
                       help='File to dump beam information to.')
    group.add_argument('-n_best', type=int, default=1,
                       help="""If verbose is set, will output the n_best
                       decoded sentences""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('-batch_size', type=int, default=8,
                       help='Batch size')
    group.add_argument('-gpu', type=int, default=0,
                       help="Device to run on")

    # Options most relevant to speech.
    group = parser.add_argument_group('Speech')
    group.add_argument('-sample_rate', type=int, default=16000,
                       help="Sample rate.")
    group.add_argument('-window_size', type=float, default=.02,
                       help='Window size for spectrogram in seconds')
    group.add_argument('-window_stride', type=float, default=.01,
                       help='Window stride for spectrogram in seconds')
    group.add_argument('-window', default='hamming',
                       help='Window type for spectrogram generation')

    # Option most relevant to image input
    group.add_argument('-image_channel_size', type=int, default=3,
                       choices=[3, 1],
                       help="""Using grayscale image can training
                       model faster and smaller""")
    

parser = argparse.ArgumentParser(
    description='translate.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

opt = translate_opts(parser)
opt = parser.parse_args()

translator = build_translator(opt, report_score=True)

@app.route('/')
def GetStatusService():
    return "start",status.HTTP_200_OK

@app.route('/abstract', methods=["POST"])
def abstract():   
    content = request.get_json()
    # print(content)
    result = {}
    result['Summary'] = ''
    if content['text'] is not None:
        # content['text'] = content['document']
        
        # list text input 
        # ex: [ document1, document2]
        assert isinstance(content['text'],  list)
        if len(content['text']) > 0:
            texts_to_translate = [preprocess(x) for x in content['text']]
            try:
                scores, predictions = translator.translate(
                    src_data_iter=texts_to_translate,
                    batch_size=opt.batch_size)
            except RuntimeError as e:
                raise ("Runtime Error: %s" % str(e))
            clean_summary =[]
            for pred in predictions:
                clean_summary.append(clean_summary_str(pred[0]))
        result['Summary'] = clean_summary
        return  result
    return result
    
app.run(host='0.0.0.0', port=6688)


