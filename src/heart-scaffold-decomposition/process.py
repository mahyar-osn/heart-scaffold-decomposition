from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field
from opencmiss.zinc.node import Node
from opencmiss.zinc.status import OK as RESULT_OK
from opencmiss.utils.zinc.general import ChangeManager
from opencmiss.utils.zinc.field import findOrCreateFieldCoordinates

import numpy as np

from sklearn import decomposition

MARKERS = [165, 224, 225, ]


def _get_nodes(field_module, coordinates, time):
    with ChangeManager(field_module):
        source_fe_field = coordinates.castFiniteElement()
        cache = field_module.createFieldcache()

        nodes = field_module.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        node_template = nodes.createNodetemplate()
        node_iter = nodes.createNodeiterator()
        node = node_iter.next()

        node_list = []

        cache.setTime(time)

        while node.isValid():
            node_template.defineFieldFromNode(source_fe_field, node)
            cache.setNode(node)

            temp = []
            if node.getIdentifier() in MARKERS:
                # print(node.getIdentifier())
                node = node_iter.next()
                continue
            for derivative in [Node.VALUE_LABEL_VALUE,
                               Node.VALUE_LABEL_D_DS1,
                               Node.VALUE_LABEL_D_DS2,
                               Node.VALUE_LABEL_D_DS3]:
                result, values = source_fe_field.getNodeParameters(cache, -1, derivative, 1, 3)
                temp.append(values)

            node_list.append(np.asarray(temp).T.tolist())

            node = node_iter.next()

    return np.asarray(node_list)


def _set_node_params():
    return


def generate_pca_scaffold(node_array, weight):
    template_context = Context("BiVentricular Model")
    template_region = template_context.getDefaultRegion()
    r = template_region.readFile(r'D:\12-labours\heart\customised-scaffold\mesh.exf')
    assert r == RESULT_OK, "Error reading Ex file!"
    template_field_module = template_region.getFieldmodule()
    template_coordinates = template_field_module.findFieldByName('coordinates')
    reference_nodes = template_field_module.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)

    with ChangeManager(template_field_module):
        source_fe_field = template_coordinates.castFiniteElement()
        cache = template_field_module.createFieldcache()
        node_template = reference_nodes.createNodetemplate()
        node_iter = reference_nodes.createNodeiterator()
        node = node_iter.next()

        counter = 0
        while node.isValid():
            node_template.defineFieldFromNode(source_fe_field, node)
            cache.setNode(node)

            if node.getIdentifier() in MARKERS:
                print(node.getIdentifier())
                node = node_iter.next()
                continue

            r = source_fe_field.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1,
                                                  node_array[counter].T[0].tolist())
            if r != RESULT_OK:
                print(f"U was not set for node {node.getIdentifier()}")
            r = source_fe_field.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1,
                                                  node_array[counter].T[1].tolist())
            if r != RESULT_OK:
                print(f"ds1 was not set for node {node.getIdentifier()}")

            r = source_fe_field.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1,
                                                  node_array[counter].T[2].tolist())
            if r != RESULT_OK:
                print(f"ds2 was not set for node {node.getIdentifier()}")

            r = source_fe_field.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1,
                                                  node_array[counter].T[3].tolist())
            if r != RESULT_OK:
                print(f"ds3 was not set for node {node.getIdentifier()}")

            counter += 1
            node = node_iter.next()

    template_region.writeFile(rf'D:\12-labours\heart\customised-scaffold\mesh_{weight}.exf')
    print()


def get_max_time(coordinates, nodes):
    nodepoint = nodes.createNodeiterator().next()
    node_template = nodes.createNodetemplate()
    field = coordinates.castFiniteElement()
    result = node_template.defineFieldFromNode(field, nodepoint)
    assert result == RESULT_OK, "Error getting time!"

    timesequence = node_template.getTimesequence(field)

    return timesequence.getNumberOfTimes()


def decompose(data):
    X = np.asarray(data.T)
    pca = decomposition.PCA(n_components=24)
    pca.fit(X)
    mean = pca.mean_
    components = pca.components_.T
    variance = pca.explained_variance_

    mode_scores = list()
    for j in range(len(X)):
        cell = X[j] - pca.mean_
        score = cell @ pca.components_.T
        mode_scores.append(score)

    _score_mean = np.average(mode_scores, axis=0)
    _score_sd = np.std(mode_scores, axis=0)

    projected_scores = list()
    for j in range(len(X)):
        cell = X[j] - pca.mean_
        score_0 = cell @ pca.components_.T
        score_z = (score_0 - _score_mean) / _score_sd  # project scores onto a Gaussian
        projected_scores.append(score_z)

    return mean, components, variance, mode_scores, projected_scores


if __name__ == '__main__':
    file_path = r'D:\12-labours\heart\beating-heart-export\beating_heart.exf'
    context = Context("BiVentricular Model")
    region = context.getDefaultRegion()
    result = region.readFile(file_path)
    assert result == RESULT_OK, "Error reading Ex file!"
    field_module = region.getFieldmodule()
    coordinates = field_module.findFieldByName('fitted coordinates')
    nodes = field_module.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    max_time = get_max_time(coordinates, nodes)

    dof_size = nodes.getSize() - len(MARKERS)

    # get nodes
    nodes_array = np.zeros((dof_size * 3 * 4, max_time))
    for t in range(max_time):
        node_values = _get_nodes(field_module, coordinates, float(t))
        nodes_array[:, t] = node_values.flatten()

    print(nodes_array.shape)
    m, c, v, score, score_norm = decompose(nodes_array)
    m_ = m.reshape(dof_size, 3, 4)

    mode_one_norm = np.asarray([i[0] for i in score_norm])
    mode_two_norm = np.asarray([i[1] for i in score_norm])

    weights = ['n2', 'n1', 'mean', 'p1', 'p2']
    count = 0
    for i in np.linspace(-2, 2, len(weights)):
        a = m + (i * c[:, 1])
        a_ = a.reshape(dof_size, 3, 4)
        generate_pca_scaffold(a_, weights[count])
        count += 1
    print('')
