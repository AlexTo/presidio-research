import argparse
import json
from collections import Counter
from typing import List

import requests

from presidio_evaluator import InputSample, ModelEvaluator
from presidio_evaluator.data_generator import read_synth_dataset
from presidio_evaluator.span_to_tag import span_to_tag, tokenize

ENDPOINT = "http://40.113.201.221:8080/api/v1/projects/test/analyze"


class PresidioAPIEvaluator(ModelEvaluator):

    def __init__(self, endpoint=None, all_fields=False, entities_to_keep=None,
                 verbose=False, labeling_scheme="IO", **kwargs):
        """
        evaluator model for the presidio API as a system
        :param endpoint: url of presidio API
        :param all_fields: boolean, true if no entities filtering should take
        place
        :param entities_to_keep: list of entities to return if found
        :param labeling_scheme: BIO/IOB or BILOU
        :param verbose:
        :param kwargs:
        """

        if not endpoint:
            print(
                "Endpoint is missing. using default presidio API at {}".format(
                    ENDPOINT))
            self.endpoint = ENDPOINT
        else:
            self.endpoint = endpoint

        if not entities_to_keep and not all_fields:
            raise ValueError("Please provide either a list of entities or"
                             "all_fields=true")

        if all_fields:
            entities_to_keep = None
        super().__init__(verbose=verbose, entities_to_keep=entities_to_keep,
                         labeling_scheme=labeling_scheme, **kwargs)

        self.set_analyze_template(all_fields=all_fields,
                                  entities=entities_to_keep)

    def predict(self, sample: InputSample):
        text = sample.full_text
        request = {"text": text,
                   "analyzeTemplate": self.analyze_template
                   }
        # Call presidio API
        r = requests.post(self.endpoint, json=request)
        starts = []
        ends = []
        tags = []

        if r.status_code == 200:
            analyzer_results = json.loads(r.text)
            if self.verbose:
                print(analyzer_results)

            if analyzer_results:
                for res in analyzer_results:
                    if not res['location'].get('start'):
                        res['location']['start'] = 0
                    starts.append(res['location']['start'])
                    ends.append(res['location']['end'])
                    tags.append(res['field']['name'])

            response_tags = span_to_tag(scheme=self.labeling_scheme,
                                        text=text,
                                        start=starts,
                                        end=ends,
                                        tag=tags)

        elif r.status_code == 400 or r.text == "":
            if self.verbose:
                print("Status 400 received")
            response_tags = ['O' for token in sample.tokens]
        else:
            print("Error getting result from Presidio API")
            print("Request = {}".format(request))
            print("Response = {}".format(r.text))
            raise Exception(r)

        return response_tags

    def set_analyze_template(self, all_fields: bool, entities: List[str]):
        template = {
            "fields": [{"name": "EMAIL_ADDRESS"}, {"name": "IP_ADDRESS"},
                       {"name": "US_DRIVER_LICENSE"},
                       {"name": "US_ITIN"}, {"name": "US_SSN"},
                       {"name": "ORG"},
                       {"name": "DOMAIN_NAME"}, {"name": "BIRTHDAY"},
                       {"name": "URL"},
                       {"name": "IBAN_CODE"}, {"name": "PERSON"},
                       {"name": "PHONE_NUMBER"},
                       {"name": "US_BANK_NUMBER"}, {"name": "CRYPTO"},
                       {"name": "NRP"},
                       {"name": "UK_NHS"},
                       {"name": "CREDIT_CARD"},
                       {"name": "DATE_TIME"},
                       {"name": "LOCATION"}, {"name": "US_PASSPORT"}]}

        if all_fields:
            self.analyze_template = template
            return

        requested_fields = []
        for entity in entities:
            for field in template['fields']:
                if entity == field['name']:
                    requested_fields.append(field)

        new_template = {'fields': requested_fields}

        self.analyze_template = new_template


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', action='store_true', default="http://localhost:8080/api/v1/projects/test/analyze")
    parser.add_argument('--dataset', type=str, default='./data/Synth/synth_test.json',
                        help='Coco path')
    args = parser.parse_args()
    # Mapping between dataset entities and Presidio entities. Key: Dataset entity, Value: Presidio entity
    entities_mapping = {
        'PERSON': 'PERSON',
        'EMAIL': 'EMAIL_ADDRESS',
        'CREDIT_CARD': 'CREDIT_CARD',
        'FIRST_NAME': 'PERSON',
        'PHONE_NUMBER': 'PHONE_NUMBER',
        'LOCATION': 'LOCATION',
        'BIRTHDAY': 'BIRTHDAY',
        'DATE': 'DATE_TIME',
        'CITY': 'LOCATION',
        'ADDRESS': 'LOCATION',
        'IBAN': 'IBAN_CODE',
        'URL': 'URL',
        'US_SSN': 'US_SSN',
        'IP_ADDRESS': 'IP_ADDRESS',
        'ORGANIZATION': 'ORG',
        'O': 'O'
    }
    presidio_fields = ['CREDIT_CARD', 'CRYPTO', 'BIRTHDAY', 'URL', 'EMAIL_ADDRESS', 'IBAN_CODE', 'ORG',
                       'IP_ADDRESS', 'NRP', 'LOCATION', 'PERSON', 'PHONE_NUMBER', 'US_SSN']

    input_samples = read_synth_dataset(args.dataset)
    new_list = ModelEvaluator.align_input_samples_to_presidio_analyzer(input_samples,
                                                                       entities_mapping,
                                                                       presidio_fields)

    flatten = lambda l: [item for sublist in l for item in sublist]

    count_per_entity_new = Counter(
        [span.entity_type for span in flatten([input_sample.spans for input_sample in new_list])])

    presidio = PresidioAPIEvaluator(all_fields=False,
                                    entities_to_keep=list(count_per_entity_new.keys()),
                                    endpoint=args.endpoint)

    evaluated_sample = presidio.evaluate_all(new_list)
    evaluation_result = presidio.calculate_score(evaluated_sample)
    evaluation_result.print()
