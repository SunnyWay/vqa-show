import random

from flask import Flask
from flask import url_for
from flask import render_template

from CocoqaDataset import CocoqaDataset
from ModelManager import ModelManager

app = Flask(__name__)

coco = CocoqaDataset()
MM = ModelManager(coco)

@app.route('/dataset/')
@app.route('/dataset/<split_name>/')
@app.route('/dataset/<split_name>/<q_type>/')
@app.route('/dataset/<split_name>/q_type>/<index>/')
def show_dataset(split_name='train', q_type='all', index=-1):
    sample = coco.get(split_name=split_name, q_type=q_type, index=index)
    return render_template('dataset.html',
                           image=url_for('static', filename=sample[0]),
                           question=sample[1],
                           answer=sample[2])

@app.route('/model/<model_name>/<split_name>/')
@app.route('/model/<model_name>/<split_name>/<q_type>/')
@app.route('/model/<model_name>/<split_name>/<q_type>/<r_type>/')
@app.route('/model/<model_name>/<split_name>/<q_type>/<r_type>/<index>/')
def show_model(model_name, split_name='train', q_type='all', r_type='all',
        index=-1):
    pred, sample = MM.get(model_name, split_name=split_name, q_type=q_type,
            r_type=r_type, index=index)
    css = [url_for('static', filename='bootstrap-combined.min.css')]
    js = [url_for('static', filename='bootstrap.min.js')]
    return render_template('model.html',
        image=url_for('static', filename=sample[0]),
        question=sample[1],
        answer=sample[2],
        pred=pred,
        css=css,
        js=js)


@app.route('/model/<model_name>/')
def model_acc(model_name):
    head = ['', 'object', 'number', 'color', 'location', 'total']
    body = [['train',], ['test',]]
    splits = ['train', 'test']
    types = ['object', 'number', 'color', 'location', 'all']
    for i, split_name in enumerate(splits):
        for q_type in types:
            body[i].append('{:.2%}'.format(MM.acc(model_name, split_name=split_name, 
                q_type=q_type)))
    css = [url_for('static', filename='bootstrap-combined.min.css')]
    js = [url_for('static', filename='bootstrap.min.js')]
    return render_template('table.html', table_name=model_name, 
            head=head, body=body, css=css, js=js)


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
