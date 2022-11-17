from flask import Flask, request, jsonify, render_template
import numpy as np
from pyspark import SparkContext
from pyspark.ml.pipeline import PipelineModel

def get_dummy(df,categoricalCols,continuousCols,labelCol,model =None):
    if model == None:
      indexers = [ StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
                  for c in categoricalCols ]

      # default setting: dropLast=True
      encoders = [ OneHotEncoder(inputCol=indexer.getOutputCol(),
                  outputCol="{0}_encoded".format(indexer.getOutputCol()))
                  for indexer in indexers ]

      assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders]
                                  + continuousCols, outputCol="features")

      pipeline = Pipeline(stages=indexers + encoders + [assembler])

      model=pipeline.fit(df)
      model.write().overwrite().save("/model/transformer")
      data = model.transform(df)
    else:
      model = PipelineModel.load(path="/model/transformer")
      data = model.transform(df)

    data = data.withColumn('label',col(labelCol))

    return data.select('features','label')


app = Flask(__name__) #Initialize the flask App

sc = SparkContext(appName="PythonLinearRegressionWithSGDExample")
model = PipelineModel.load(path="/model/gbt_model")

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict/<value_list>',methods=['POST'])
def predict(value_list):
    '''
    Example Input Array
    "51,technician,single,professional.course,no,no,no,cellular,aug,wed,1740,1,999,0,nonexistent,1.4,93.444,-36.1,4.964,5228.1,no"
    '''
    arr_value = []
    values = value_list.split(',')
    for value in values:
        arr_value.append(int(value))

    df = pd.DataFrame(arr_value,columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx',
       'cons_conf_idx', 'euribor3m', 'nr_employed', 'y'])

    a = model.transform(get_dummy(spark.createDataFrame(df), catcols, num_cols, labelCol,model='pretrained'))
    prediction_result = a.first().asDict()['predictedLabel']
    return jsonify(
        value_input=value_list,
        result=prediction_result
    )
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')