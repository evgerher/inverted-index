from collections import defaultdict
import re
import sys
from io import TextIOWrapper
from typing import Iterable, Dict, Tuple, List, TextIO, Optional, Set
from argparse import FileType, ArgumentParser, ArgumentTypeError

from packing import StoragePolicy


class EncodedFileType(FileType):
    def __call__(self, string):
        if string == '-':
            if 'r' in self._mode:
                stdin = TextIOWrapper(sys.stdin.buffer, encoding=self._encoding)
                return stdin.readlines()
            elif 'w' in self._mode:
                stdout = TextIOWrapper(sys.stdout.buffer, encoding=self._encoding)
                return stdout
            else:
                msg = 'arguments "-" with mode %r' % self._mode
                raise ValueError(msg)

        try:
            return open(string, self._mode, self._bufsize, self._encoding, self._errors)
        except OSError as e:
            message = "can't open '%s': %s"
            raise ArgumentTypeError(message % (string, e))


class InvertedIndex:
    def __init__(self, word2doc_id: Dict[str, Set[int]]):
        self._word2doc_id = word2doc_id

    def query(self, query: List[str]) -> List[int]:
        '''
        Query inverted index against a collection of words
        :param query: list of words
        :return: intersection of doc_ids for all words
        '''
        words_present = [w for w in query if w in self._word2doc_id.keys()]
        if len(words_present) == 0:
            return []
        main_set = self._word2doc_id[words_present[0]]
        for word in words_present:
            word_set = self._word2doc_id[word]
            main_set = main_set.intersection(word_set)
        return list(main_set)

    def dump(self, filepath: str, storage_policy: StoragePolicy):
        '''
        Dump instance of InvertedIndex into file
        :param filepath: future stored index file path
        :param storage_policy: to apply
        :return:
        '''
        mapping = {x: list(y) for x, y in self._word2doc_id.items()}
        storage_policy.dump(mapping, filepath)

    @classmethod
    def load(cls, filepath, storage_policy: StoragePolicy):
        index = {x: set(y) for x, y in storage_policy.load(filepath).items()}
        return cls(index)

    def __eq__(self, other):
        return self._word2doc_id == other._word2doc_id


def load_documents(filepath: str, encoding: str = 'utf-8') -> Iterable[Tuple[int, str]]:
    '''
    Method returns iterable over lines found in file (`filepath`)
    :param filepath: to iterate over lines of
    :return: generator of (article_id, title, text)
    '''
    regex = r'(\d+)\s(.+)'
    with open(filepath, 'r', encoding=encoding) as f:
        for line in f:
            matches = list(re.finditer(regex, line.rstrip()))
            if len(matches) == 1:
                article_id, content = matches[0].groups()
                yield (int(article_id), content)
            else:
                raise ValueError("Unable to parse line")


def build_inverted_index(documents: Iterable[Tuple[int, str]]) -> InvertedIndex:
    '''
    Method (1) builds inverted index over words/documents and (2) initializes InvertedIndex object
    :param documents: iterable over articles found in dataset (article_id, title, text)
    :return: InvertedIndex instance
    '''
    word2doc_id: Dict[str, Set[int]] = defaultdict(set)

    for doc_id, content in documents:
        words = content.split()
        for word in words:
            word2doc_id[word].add(doc_id)

    return InvertedIndex(word2doc_id)


def create_index(dataset_filepath: str, output_filepath: str, storage_policy_name: str = 'struct'):
    '''
    CLI method for index creation from scratch
    :param dataset_filepath: to process articles from
    :param output_filepath: to store index
    :param storage_policy_name: to apply ('json' or 'struct')
    '''
    doc_gen = load_documents(dataset_filepath)
    storage_policy = StoragePolicy.create(storage_policy_name)
    index = build_inverted_index(doc_gen)
    index.dump(output_filepath, storage_policy)


def process_query(index_filepath: str,
                  query_file: Optional[TextIO] = None,
                  queries: List[List[str]] = None,
                  storage_policy_name: str = 'struct'):
    '''
    CLI method for query processing
    :param index_filepath: to load index from
    :param query_file: file-provided queries (query per line)
    :param queries: cli-provided queries
    :param storage_policy_name: to unpack index with ('struct' or 'json')
    :return:
    '''
    def output_result(out: List[int]):
        out = [str(x) for x in out]
        print(','.join(out), end='\n')

    storage_policy = StoragePolicy.create(storage_policy_name)
    index = InvertedIndex.load(index_filepath, storage_policy)
    if queries is not None:
        for subquery in queries:
            idxs = index.query(subquery)
            output_result(idxs)
    else:
        for line in query_file:
            words = line.rstrip().split(' ')
            idxs = index.query(words)
            output_result(idxs)


def callback_build(arguments):
    '''
    Callback wrapper for create_index
    :param arguments:
    :return:
    '''
    create_index(arguments.dataset_filepath, arguments.output_filepath, arguments.storage_policy)


def callback_query(arguments):
    '''
    Callback wrapper for process_query
    :param arguments:
    :return:
    '''
    process_query(arguments.index_filepath, arguments.query_file, arguments.query, arguments.storage_policy)


def parse_arguments():
    parser = ArgumentParser(
        prog='inverted-index',
        description='App to build inverted index over dataset & query requests to retrieve document ids'
    )

    subparsers = parser.add_subparsers(help='Choose command to use')

    # Build command
    build_parser = subparsers.add_parser('build', help="Build inverted index and store on disk")
    build_parser.add_argument('--dataset',
                              type=str,
                              dest='dataset_filepath',
                              metavar='/path/to/dataset',
                              required=True)
    build_parser.add_argument('--output',
                              type=str,
                              dest='output_filepath',
                              metavar='/path/for/index',
                              required=True)
    build_parser.add_argument('--storage_policy',
                              type=str,
                              dest='storage_policy',
                              default='struct',
                              required=False,
                              help='Select compression algorithm to dump index: "json" or "struct"')
    build_parser.set_defaults(callback=callback_build)

    ### Query command
    query_parser = subparsers.add_parser('query', help="Query inverted index and return document ids")
    query_parser.add_argument('--index',
                              required=True,
                              dest='index_filepath',
                              metavar='/path/to/index',
                              type=str)
    query_parser.add_argument('--storage_policy',
                              type=str,
                              dest='storage_policy',
                              default='struct',
                              required=False,
                              help='Select compression algorithm to load index: "json" or "struct"')

    query_group = query_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--query',
                             metavar='WORD',
                             type=str,
                             action='append',
                             help='Text query',
                             nargs='+')

    file_group = query_group.add_mutually_exclusive_group(required=False)
    file_group.add_argument('--query-file-utf8',
                            dest='query_file',
                            metavar='/path/to/file',
                            default=TextIOWrapper(sys.stdin.buffer, encoding='utf-8'),
                            type=EncodedFileType('r', encoding='utf-8'),
                            help='Path to file with queries (per line); encoding=utf-8')
    file_group.add_argument('--query-file-cp1251',
                            dest='query_file',
                            metavar='/path/to/file',
                            default=TextIOWrapper(sys.stdin.buffer, encoding='cp1251'),
                            type=EncodedFileType('r', encoding='cp1251'),
                            help='Path to file with queries (per line); encoding=cp-1251')
    query_parser.set_defaults(callback=callback_query)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    arguments.callback(arguments)


if __name__ == '__main__':
    main()
