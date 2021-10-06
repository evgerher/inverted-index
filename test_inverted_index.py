import os
from collections import Counter
import json

import pytest

from task_sorokin_evgeny_inverted_index import load_documents, build_inverted_index, InvertedIndex
from packing import JsonStoragePolicy, StructStoragePolicy


@pytest.fixture
def empty_document(tmpdir_factory):
    folder = tmpdir_factory.mktemp("data")
    fn = folder.join("article.txt")
    fn.write_text('', encoding='utf-8')  # for empty file to be initialized
    return fn


@pytest.fixture
def single_document_example(empty_document):
    text = "1337	Merol   Lorem ipsum ist dalor\n"
    empty_document.write_text(text, encoding='utf-8')
    return empty_document


@pytest.fixture
def multiple_document_example(empty_document):
    text = "2019	title Nothing\n1984	Consensus 2 + 2 = 5\n"
    empty_document.write_text(text, encoding='utf-8')
    return empty_document


@pytest.fixture
def generate_articles():
    return [(10, 'Title', 'Some text')]


def test_load_documents_single(single_document_example):
    items = list(load_documents(single_document_example, encoding='utf-8'))
    assert len(items) == 1

    expected_doc_id = 1337
    doc_id, text = items[0]

    assert isinstance(doc_id, int), 'Type of doc_id should be `int`'
    assert doc_id == expected_doc_id, f'Document_id should be {expected_doc_id} , while parsed {doc_id}'
    assert text == 'Merol   Lorem ipsum ist dalor'


def test_load_documents_error(empty_document):
    empty_document.write_text('1337Dog	Dog', encoding='utf-8')
    with pytest.raises(ValueError):
        list(load_documents(empty_document, encoding='utf-8'))


def test_load_documents_empty(empty_document):
    items = list(load_documents(empty_document, encoding='utf-8'))
    assert len(items) == 0


def test_load_documents_multiple(multiple_document_example):
    items = list(load_documents(multiple_document_example, encoding='utf-8'))
    assert len(items) == 2, 'Two articles present (per line) in the document should be read'

    expected_doc_id1 = 2019; expected_doc_id2 = 1984
    doc_id1, content1 = items[0]
    doc_id2, content2 = items[1]

    assert isinstance(doc_id1, int), 'Type of doc_id should be `int`'
    assert doc_id1 == expected_doc_id1, f'Document_id should be {expected_doc_id1} , while parsed {doc_id1}'
    assert content1 == 'title Nothing'

    assert isinstance(doc_id2, int), 'Type of doc_id should be `int`'
    assert doc_id2 == expected_doc_id2, f'Document_id should be {expected_doc_id2} , while parsed {doc_id2}'
    assert content2 == 'Consensus 2 + 2 = 5'


def test_build_inverted_index_empty():
    index = build_inverted_index([])
    assert len(index._word2doc_id) == 0
    assert isinstance(index, InvertedIndex), f'Return type of `build_inverted_index` should be InvertedIndex, ' \
                                             f'found {type(index)}'


def test_build_inverted_index_single():
    doc_id, content = 10, 'Doctor Who		John Wick was right'
    expected_number_of_words = 6

    index = build_inverted_index([(doc_id, content)])

    assert len(index._word2doc_id) == expected_number_of_words
    assert all([len(v) for v in index._word2doc_id.values()]), "All word2doc_id lists should be " \
                                                               "length of 1 (only one doc present)"
    assert all([list(v)[0] == doc_id for v in index._word2doc_id.values()]), f"Only doc_id = {doc_id} should be present"


def test_build_inverted_index_same_words():
    doc_id, content = 10, 'Smoken Albion	Smoken Albion'
    expected_number_of_words = 2

    index = build_inverted_index([(doc_id, content)])

    assert len(index._word2doc_id) == expected_number_of_words


def test_build_inverted_index_case_sensitive():
    doc_id, content = 10, 'salmon Salmon'
    expected_number_of_words = 2
    expected_words = ['salmon', 'Salmon'].sort()

    index = build_inverted_index([(doc_id, content)])
    index_words = list(index._word2doc_id.keys()).sort()
    n_words = len(index._word2doc_id)

    assert n_words == expected_number_of_words, f"Built index contains {n_words}, " \
                                                f"it should match expected {expected_number_of_words}"
    assert expected_words == index_words, f"Index words ({index_words}) " \
                                          f"should be the same as expected ({expected_words})"


def test_inverted_index_contain_no_repeated_idxs():
    articles = [
        (12, 'A B C A'),
        (13, 'A B C D'),
        (14, 'B'),
    ]
    index = build_inverted_index(articles)
    mapping = index._word2doc_id

    counterA = Counter(mapping['A'])
    counterB = Counter(mapping['B'])
    assert len(mapping['A']) == 2 and len(mapping['B']) == 3
    for counter in [counterA, counterB]:
        assert all(x == 1 for x in counter.values()), f"Only one example of unique document should be present in index, " \
                                                      f"found={list(counter.items())}"


@pytest.mark.parametrize(
    ['query', 'etalon_answer'],
    [
        pytest.param(['A'], [1, 2, 3], id='A'),
        pytest.param(['B'], [2, 3, 4], id='B'),
        pytest.param(['A', 'B'], [2, 3], id='`A B`'),
        pytest.param(['C'], [], id='Unknown word'),
    ],
)
def test_inverted_index_query_intersection(query, etalon_answer):
    mapping = {'A': {1, 2, 3}, 'B': {2, 3, 4}}
    index = InvertedIndex(mapping)
    idxs = index.query(query)
    assert sorted(idxs) == sorted(etalon_answer), f'Expected answer is {etalon_answer}, but got {idxs}'


def test_inverted_index_query_no_intersection():
    mapping = {'A': {1, 2}, 'B': {3, 4}}
    index = InvertedIndex(mapping)

    idxs = index.query(['A'])
    assert all([x in mapping['A'] and x not in mapping['B'] for x in idxs])

@pytest.mark.parametrize(
    ['storage_policy'],
    [
        pytest.param(JsonStoragePolicy, id='json policy'),
        pytest.param(StructStoragePolicy, id='struct policy'),
    ]
)
def test_inverted_index_dump_load_equal(tmpdir, storage_policy):
    mapping = {'A': {1, 2, 3}, 'B': {2, 3, 4}}
    index = InvertedIndex(mapping)
    fn = tmpdir.join('dump.index')

    index.dump(fn.strpath, storage_policy)
    loaded_index = InvertedIndex.load(fn.strpath, storage_policy)

    assert index == loaded_index, "Dumped and loaded index should be equal"


from task_sorokin_evgeny_inverted_index import create_index, process_query, build_inverted_index, EncodedFileType
from packing import JsonStoragePolicy


@pytest.fixture
def index_filepath(tmpdir):
    collection = [
        (1, 'A\tThis text is made for test'),
        (2, 'B\t\tAnother article contains A'),
        (3, 'C Any word , no repeatition'),
        (4, 'D  Sudden русский язык'),
    ]
    index = build_inverted_index(collection)
    index_file = tmpdir.join('tmp.index')
    index_filepath = index_file.strpath
    index.dump(index_filepath, JsonStoragePolicy)
    return index_filepath

@pytest.mark.parametrize(
    ['storage_policy_name'],
    [
        pytest.param('json', id='json policy'),
        pytest.param('struct', id='struct policy'),
    ]
)
def test_cli_create_index_file(tmpdir, storage_policy_name):
    dataset_file = tmpdir.join('test_dataset.txt')
    dataset_file.write_text('101  Aloha  Hans bring ze flammenwerfer\n', encoding='utf-8')
    dataset_filepath = dataset_file.strpath

    output_filepath = tmpdir.strpath + '/tmp.index'
    create_index(dataset_filepath, output_filepath, storage_policy_name=storage_policy_name)

    assert os.path.isfile(output_filepath), f"`create_index` should create file {output_filepath}"


@pytest.mark.parametrize(
    ['query', 'etalon_answer'],
    [
        pytest.param(['A'], '1,2', id='A'),
        pytest.param(['B'], '2', id='B'),
        pytest.param(['A', 'B'], '2', id='`A B`'),
        pytest.param(['C', 'B'], '', id='No intersection'),
    ],
)
def test_cli_query_text(index_filepath, query, etalon_answer, capsys):
    process_query(index_filepath, queries=[query], storage_policy_name='json')

    captured = capsys.readouterr()
    assert etalon_answer in captured.out, f"Expected '{etalon_answer}' in '{captured.out}'"


def test_cli_query_test_multiple(index_filepath, capsys):
    queries = [['A', 'B'], ['A']]
    expected = ['2', '1,2']
    process_query(index_filepath, queries=queries, storage_policy_name='json')

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines == expected, f"The order of answers={lines} should match expected={expected}"


@pytest.mark.parametrize(
    ['encoding'],
    [
        pytest.param('utf-8', id='utf-8'),
        pytest.param('utf-16', id='utf-16'),
        pytest.param('koi8-r', id='koi8-r'),
        pytest.param('cp1251', id='cp1251'),
    ]
)
def test_cli_query_file_encoded(index_filepath, tmpdir, capsys, encoding):
    query_encoded_tmpfile = tmpdir.join('tmpquery.txt')
    query_encoded_tmpfile.write_text('made test\nword ,\n', encoding=encoding)

    encoded_file = EncodedFileType('r', encoding=encoding)(query_encoded_tmpfile)
    process_query(index_filepath, query_file=encoded_file, storage_policy_name='json')

    expected = ['1', '3']
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines == expected


@pytest.mark.parametrize(
    ['storage_policy_name'],
    [
        pytest.param('json', id='json policy'),
        pytest.param('struct', id='struct policy'),
    ]
)
def test_cli_create_and_query_text(tmpdir, capsys, storage_policy_name):
    dataset_file = tmpdir.join('test_dataset.txt')
    dataset_file.write_text('101  Lorem  Ipsum sit dalor\n3  Ipsum  Lorem ipsum dalor sit amet\n', encoding='utf-8')
    dataset_filepath = dataset_file.strpath

    output_filepath = tmpdir.strpath + '/tmp.index'
    create_index(dataset_filepath, output_filepath, storage_policy_name=storage_policy_name)

    queries = [['Lorem', 'sit']]
    process_query(output_filepath, queries=queries, storage_policy_name=storage_policy_name)

    captured = capsys.readouterr()
    assert '3,101' in captured.out


@pytest.mark.parametrize(
    ['mapping', 'etalon_answer'],
    [
        pytest.param({},
                     '{}',
                     id='empty index'),
        pytest.param({'word': [10, 20, 30], 'kelvin': [20]},
                     '{"word": [10, 20, 30], "kelvin": [20]}',
                     id='Index with 2 words'),
    ],
)
def test_json_storage_dump(tmpdir, mapping, etalon_answer):
    file = tmpdir.join('index.dump')
    JsonStoragePolicy.dump(mapping, file.strpath)

    with open(file, 'r') as f:
        content = f.read()
        assert content == etalon_answer, f"Expected file content {etalon_answer}, but found {content}"

    index = JsonStoragePolicy.load(file.strpath)
    assert index == mapping, f'Loaded {mapping} and dumped json objects should be equal, found {index}'


def test_json_load_fail(tmpdir):
    file = tmpdir.join('index.dump')
    invalid_line = '{"doctor": [10], '
    file.write_text(invalid_line, 'utf-8')

    with pytest.raises(json.JSONDecodeError):
        JsonStoragePolicy.load(file.strpath)


def test_struct_dump_empty(tmpdir):
    file = tmpdir.join('index.dump')
    mapping = {}
    bytes_written = StructStoragePolicy.dump(mapping, file.strpath)
    assert bytes_written == 4, 'Only single integer is written into file, the size should be 4'


@pytest.mark.parametrize(
    ['mapping'],
    [
        pytest.param({}, id='empty index'),
        pytest.param({'word': (10, 20, 30), 'kelvin': (20,)},
                     id='Index with 2 words'),
    ],
)
def test_struct_dump(tmpdir, mapping):
    file = tmpdir.join('index.dump')
    bytes_written = StructStoragePolicy.dump(mapping, file.strpath)

    loaded_index = StructStoragePolicy.load(file.strpath)
    assert loaded_index == mapping, f"Loaded {loaded_index} and initial {mapping} mapping should be equal"

