
import os
from py2neo import Graph, Node, Relationship, NodeMatcher

def find_files(directory, filetype='docx'):
    docx_files = []
    sub_paths = []

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(f".{filetype}"):
                docx_files.append(filename)
                sub_paths.append(dirpath)

    return docx_files, sub_paths


def store_triples_mutipletimes_into_neo4j(text_file, first_flag):
    work_path = os.path.abspath('.') + '/neo4j_store/'
    if not os.path.exists(work_path + text_file):
        ValueError('Triples file {0} does not exist'.format(work_path + text_file))
    else:
        graph = Graph(
            "bolt://localhost:7687", 
            auth=("neo4j", "123456789")
        )
        if first_flag == True:
            graph.delete_all()
        with open(work_path + text_file, 'r+', encoding='utf-8') as file:
            line = file.readline()
            replace_list=['digital twins', 'Digital Twins', 'digital twin', 'Digital twin', 'Digital Twin', 'Digital Twinss', 'digital-twin', 'a digital twin']
            while line:
                clear_line = line.strip('\n')
                # judge string is start with '[' and end with ']'
                if clear_line.startswith('[') and clear_line.endswith(']'):
                    # strip start and end character 
                    clear_line = clear_line.lstrip('[').rstrip(']')
                    triple = clear_line.split(';')

                    # triples len is longer than 3, then continue
                    if not len(triple) == 3:
                        line = file.readline()
                        continue
                    # get the triple's sunject
                    triple_subject = triple[0]
                    # replace same meaning subject
                    if triple_subject in replace_list:
                        triple_subject = 'Digital twins'
                    if triple_subject == 'artificial intelligence':
                        triple_subject = 'AI'
                    # get the triple's relation
                    triple_relation = triple[1].lstrip()
                    # get the triple's object
                    triple_object = triple[2].lstrip()
                    # replace same meaning object
                    if triple_object in replace_list:
                        triple_object = 'Digital twins'
                    if triple_object == 'artificial intelligence':
                        triple_object = 'AI'

                    # add triples into graph
                    matcher = NodeMatcher(graph)
                    subject_list = list(matcher.match('node', name=triple_subject))
                    object_list = list(matcher.match('node', name=triple_object))
                    if len(subject_list) > 0:
                        # subject node has already exist
                        if len(object_list) > 0:
                            # subject & object both exist
                            relation = Relationship(subject_list[0], triple_relation, object_list[0])
                            graph.create(relation)
                        else:
                            # only subject exist, object does not exist
                            object_node = Node('node', name=triple_object)
                            relation = Relationship(subject_list[0], triple_relation, object_node)
                            graph.create(relation)
                    else:
                        # subject does not exist
                        if len(object_list) > 0:
                            # only object exist, subject does not exist
                            subject_node = Node('node', name=triple_subject)
                            relation = Relationship(subject_node, triple_relation, object_list[0])
                            graph.create(relation)
                        else:
                            # subject & object both do not exist
                            subject_node = Node('node', name=triple_subject)
                            object_node = Node('node', name=triple_object)
                            relation = Relationship(subject_node, triple_relation, object_node)
                            graph.create(relation)
                    line = file.readline()
                else:
                    line = file.readline()

if __name__ == '__main__':
    work_path = os.path.abspath('.') + '/neo4j_store'
    txt_files, sub_paths = find_files(work_path, filetype='txt')
    first_flag = True
    for txt_file in txt_files:
        if not txt_file == 'requirements.txt':
            print('Write {0} file triples into neo4j'.format(txt_file))
            store_triples_mutipletimes_into_neo4j(txt_file, first_flag)
            first_flag = False