from bokeh.core.enums import SizingMode
from bokeh.models.annotations import Title
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import sqlalchemy as db
import pandas as pd
import json
from bokeh.models import ColumnDataSource
from bokeh.models import DataTable, TableColumn, Div
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from bokeh.models import ColumnDataSource, MultiLine, Line, Legend, LegendItem
from bokeh.plotting import figure
from bokeh.palettes import Viridis256, Spectral11, Set3_12, Category10_10
from bokeh.io import show, output_notebook, curdoc
from bokeh.layouts import column, gridplot, layout
import os
import collections
'''
def json_model_data(predictions):
    pred_json = json.loads(predictions)

    disallowed_characters = '[]"'
    for key in pred_json:
        for character in disallowed_characters:
            pred_json[key] = pred_json[key].replace(character, '')
            pred_json[key] = pred_json[key].split(',')[0]

    #pred_df = pd.DataFrame.from_dict(pred_json, orient='index', columns=['Model Prediction'])
    pred_df = pd.DataFrame(pred_json)
    pred_df.index = pred_df.index.rename("Model ID")
    prediction_df = pred_df.sort_values(by=['Model ID'])
    #prediction_df = prediction_df.reset_index()
    prediction_df = prediction_df.transpose()
    prediction_df = prediction_df.reset_index(drop=True) #.drop('index', axis=1)
    return prediction_df
'''

def get_model_info(model_id, attribute, connection):
    query = "SELECT " + str(attribute) + " from models WHERE id="+str(model_id)
    model_name = connection.execute(query).first()[0]
    return model_name

# Calculate accuracy by day
def grouped_score(true_labels, predictions, score):
    #true_labels = mlb.fit_transform(true_labels.apply(lambda x: x.split(',')))
    predictions = mlb.fit_transform(predictions.apply(lambda x: x.split(',')))
    true_labels = mlb.fit_transform(true_labels)
    #predictions = mlb.fit_transform(predictions)
    if score=='accuracy':
        metric = accuracy_score(true_labels, predictions)
    elif score=='precision':
        metric = precision_score(true_labels, predictions, average='weighted')
    elif score=='recall':
        metric = recall_score(true_labels, predictions, average='weighted')
    elif score=='f1':
        metric = f1_score(true_labels, predictions, average='weighted')
    return metric

# This identifies the rows to use for multi-class predictions, since they are only trained when there is 1 ship present
# This will create an accuracy score to use for only the instances with 1 ship present
def multi_class_indicator(true_labels):
    #true_label = true_labels.split(',')
    if len(true_labels) == 1:
        return True
    elif len(true_labels) > 1:
        return False

# If multiclass (single class prediction) is in the list of true labels, count as correct prediction
def subset_multiclass(true_label, prediction):
    if prediction in true_label: #.split(','):
        return prediction
    else:
        return true_label

def json_model_data(predictions):
    pred_json = json.loads(predictions)
    pred_df = pd.DataFrame(pred_json)
    pred_df=pred_df.reset_index(drop=True)
    pred_df = pd.DataFrame(pred_df.iloc[0]).T

    return pred_df

app_db = os.environ.get('DATABASE_URL')

app_engine = db.create_engine(app_db)
app_connection = app_engine.connect()

query = "SELECT * FROM PREDICTIONS WHERE true_label IS NOT NULL" #"SELECT * FROM PREDICTIONS"

#SELECT * FROM `PREDICTIONS` WHERE true_label IS NOT NULL AND TRIM(true_label) <> ''

predictions = pd.read_sql_query(query, app_connection)

if not predictions.empty:
    # Need to repeat this process for every radius, e.g. for x in radius_list
    radius=20

    accuracy_df = pd.DataFrame()
    for index, row in predictions.iterrows():
        models = json_model_data(row['model_predictions'])
        #models = models.squeeze()
        #append_data = row[['start_time', 'end_time', 'true_label']]

        #append_data['true_label'] = str(json.loads(append_data['true_label'])[str(radius)])
        #append_data['true_label'] = json.loads(append_data['true_label'])[str(radius)]
        #models['true_label_multi_instance'] = ",".join(json.loads(row['true_label'])[str(radius)])
        #models['true_label_multi_label'] = ",".join(set(json.loads(row['true_label'])[str(radius)]))
        #print(models)\

        # Skip rows where there is no true label because ship is in the ambiguous zone between range and no ship identification
        if None in set(json.loads(row['true_label'])[str(radius)]):
            #print(set(json.loads(row['true_label'])[str(radius)]))
            continue
        else:
            #print(set(json.loads(row['true_label'])[str(radius)]))
            models['true_label_multi_label'] = [set(json.loads(row['true_label'])[str(radius)])]
            models['true_label_multi_instance'] = [json.loads(row['true_label'])[str(radius)]]

        

        '''
        disallowed_characters = '[]"'
        for character in disallowed_characters:
            append_data['true_label'] = append_data['true_label'].replace(character, '')
        '''

        start_time = row['start_time']
        end_time = row['end_time']

        models['start_time'] = start_time
        models['end_time'] = end_time
        
        #tmp_accuracy = models.append(append_data)
        '''
        if accuracy_df.empty:
            accuracy_df = pd.DataFrame([models])
        else:
            accuracy_df = pd.concat([accuracy_df, pd.DataFrame([models])])
        '''
        accuracy_df = pd.concat([accuracy_df, models], ignore_index=True)

    # Drop instances where the true label is none

    #models = models[models['true_label_multi_label'].notna]

    # Get rid of instances with unknown classes
    #accuracy_df = accuracy_df[accuracy_df["true_label"].str.contains("Unknown")==False]

    # Get unique class values from true labels, This is for multi_label
    #accuracy_df['unique_classes'] = accuracy_df['true_label'].apply(lambda x: str(list(set(x.split(','))))[1:-1])
    #accuracy_df['unique_classes'] = accuracy_df['unique_classes'].str.replace(" '","")
    #accuracy_df['unique_classes'] = accuracy_df['unique_classes'].str.replace("'","")


    #accuracy_df = accuracy_df.reset_index().drop('index', axis=1)

    # Identifies which rows only have one ship present
    #accuracy_df['multi_class_indicator'] = accuracy_df.apply(lambda x: multi_class_indicator(x['true_label']), axis=1)
    accuracy_df['multi_class_indicator'] = accuracy_df.apply(lambda x: multi_class_indicator(x['true_label_multi_instance']), axis=1)

    # If a multi-instance, multi-label model is ever trained, use the column 'true_label' 
    # as is to compute accuracy metrics


    cols = accuracy_df.columns
    cols = list(cols)
    cols.remove('start_time')
    cols.remove('end_time')
    #cols.remove('true_label')
    cols.remove('true_label_multi_instance')
    cols.remove('true_label_multi_label')
    cols.remove('multi_class_indicator')
    #cols.remove('unique_classes')


    # Need to add accuracy scores for all data, instead of just ones that match
    # Need to add precision, recall, F1 score
    # Need to add other model identifying information into table
    # Need to add accuracy grouped over days for multiline plot
    # Add accuracy calculation for each class
    # Drop Unknown class types

    mlb = MultiLabelBinarizer(classes=['Class A', 'Class B', 'Class C', 'Class D', 'Class E'])

    #true_labels = mlb.fit_transform(accuracy_df['true_label'].apply(lambda x: x.split(',')))
    #unique_classes = mlb.fit_transform(accuracy_df['unique_classes'].apply(lambda x: x.split(',')))
    #single_ship_unique_classes = mlb.fit_transform(accuracy_df[accuracy_df['multi_class_indicator']==True]['unique_classes'].apply(lambda x: x.split(',')))

    #true_labels = mlb.fit_transform(accuracy_df['true_label_multi_instance'])
    #unique_classes = mlb.fit_transform(accuracy_df['true_label_multi_label'])
    #single_ship_unique_classes = mlb.fit_transform(accuracy_df[accuracy_df['multi_class_indicator']==True]['true_label_multi_label'])



    #accuracy_df['true_label'] = mlb.fit_transform(accuracy_df['true_label'].apply(lambda x: x.split(','))).tolist()
    #accuracy_df['unique_classes'] = mlb.fit_transform(accuracy_df['unique_classes'].apply(lambda x: x.split(','))).tolist()



    accuracy_df['start_time'] = pd.to_datetime(accuracy_df['start_time'], unit='s')
    accuracy_df['end_time'] = pd.to_datetime(accuracy_df['end_time'], unit='s')
    accuracy_df['date'] = accuracy_df['start_time'].apply(lambda x: x.date())
    accuracy_df['date'] = pd.to_datetime(accuracy_df['date'])

    daily_accuracy={}
    daily_precision={}
    daily_recall={}
    daily_f1={}
    accuracy_scores = {}
    total_accuracy_scores = {}
    classification_report_dict = {}
    # Subset of dataframe where only one ship present in the dataset
    #single_ship_accuracy_df = accuracy_df[accuracy_df['multi_class_indicator']==True]
    for column in cols:
        # Need to drop NA values from each column, because each model wasn't necessarily activate the entire time

        new_accuracy_df = accuracy_df[accuracy_df[column].notna()]

        true_labels = mlb.fit_transform(new_accuracy_df['true_label_multi_instance'])
        unique_classes = mlb.fit_transform(new_accuracy_df['true_label_multi_label'])
        single_ship_unique_classes = mlb.fit_transform(new_accuracy_df[new_accuracy_df['multi_class_indicator']==True]['true_label_multi_label'])
        single_ship_accuracy_df = new_accuracy_df[new_accuracy_df['multi_class_indicator']==True]

        predictions = mlb.fit_transform(new_accuracy_df[column].apply(lambda x: x.split(',')))
        single_ship_predictions = mlb.fit_transform(new_accuracy_df[new_accuracy_df['multi_class_indicator']==True][column].apply(lambda x: x.split(',')))

        # Use these when predictions in dictionary are not in list, need to change to below commented
        #predictions = mlb.fit_transform(accuracy_df[column].apply(lambda x: x.split(',')))
        #single_ship_predictions = mlb.fit_transform(accuracy_df[accuracy_df['multi_class_indicator']==True][column].apply(lambda x: x.split(',')))

        # Use these when predictions are in list
        #predictions = mlb.fit_transform(accuracy_df[column])
        #single_ship_predictions = mlb.fit_transform(accuracy_df[accuracy_df['multi_class_indicator']==True][column])

        # Look up model class in database to determine what column to use for metrics
        class_query = "SELECT model_type FROM models WHERE id="+str(column)

        model_class = app_connection.execute(class_query)    
        model_class = model_class.first()[0]

        
        if model_class=='multi_class':
            # If the prediction is a subset of the true label, then put that in new column
            # Will count that as a correct prediction (to account for the fact it can only predict one class)
            #accuracy_df['working_predictions'] = accuracy_df.apply(lambda x: subset_multiclass(x['unique_classes'], x[column]), axis=1)
            #working_predictions = mlb.fit_transform(accuracy_df['working_predictions'].apply(lambda x: x.split(','))) 
            #acc_score = accuracy_score(working_predictions, predictions)    

            # Use to discard instances with multiple ships since those are not used in training
            #acc_score = accuracy_score(accuracy_df[accuracy_df['multi_class_indicator']==True]['unique_classes'], accuracy_df[accuracy_df['multi_class_indicator']==True][column])

            # Calculate accuracy, precision, recall, f1 score
            
            acc_score = accuracy_score(single_ship_unique_classes, single_ship_predictions)
            prec_score = precision_score(single_ship_unique_classes, single_ship_predictions, average='weighted')
            rec_score = recall_score(single_ship_unique_classes, single_ship_predictions, average='weighted')
            fone_score = f1_score(single_ship_unique_classes, single_ship_predictions, average='weighted')

            report = classification_report(single_ship_unique_classes, single_ship_predictions, output_dict=True)

            # Calculate same scores by day
            grouped_accuracy = single_ship_accuracy_df.groupby(pd.Grouper(key='date',freq='D'))
            grouped_accuracy_days = grouped_accuracy.apply(lambda x: grouped_score(x['true_label_multi_label'], x[column], 'accuracy'))

            grouped_precision_days = grouped_accuracy.apply(lambda x: grouped_score(x['true_label_multi_label'], x[column], 'precision'))
            grouped_recall_days = grouped_accuracy.apply(lambda x: grouped_score(x['true_label_multi_label'], x[column], 'recall'))
            grouped_f1_days = grouped_accuracy.apply(lambda x: grouped_score(x['true_label_multi_label'], x[column], 'f1'))

        # for multi class, if predicted ship is in list of multiple ships, then count as accurate prediction    

        # Need to edit and make into multi_label accuracy score calculator
        elif model_class == 'multi_label':

            #convert multilabel columns to list representation, prep for accuracy function

            #acc_score = accuracy_score(accuracy_df['unique_classes'], accuracy_df[column])

            # Calculate accuracy, precision, recall, f1 score
            acc_score = accuracy_score(unique_classes, predictions)
            pred_score = precision_score(unique_classes, predictions, average='weighted')
            rec_score = recall_score(unique_classes, predictions, average='weighted')
            fone_score = f1_score(unique_classes, predictions, average='weighted')

            report = classification_report(single_ship_unique_classes, single_ship_predictions, output_dict=True)

            # Calculate same scores by day
            grouped_accuracy = new_accuracy_df.groupby(pd.Grouper(key='date',freq='D'))
            grouped_accuracy_days = grouped_accuracy.apply(lambda x: grouped_score(x['true_label_multi_label'], x[column], 'accuracy'))

            grouped_precision_days = grouped_accuracy.apply(lambda x: grouped_score(x['true_label_multi_label'], x[column], 'precision'))
            grouped_recall_days = grouped_accuracy.apply(lambda x: grouped_score(x['true_label_multi_label'], x[column], 'recall'))
            grouped_f1_days = grouped_accuracy.apply(lambda x: grouped_score(x['true_label_multi_label'], x[column], 'f1'))


        daily_accuracy[column] = grouped_accuracy_days
        daily_precision[column] = grouped_precision_days
        daily_recall[column] = grouped_recall_days
        daily_f1[column] = grouped_f1_days
        accuracy_scores[column] = [acc_score, prec_score, rec_score, fone_score]
        classification_report_dict[column] = report

        # Compute total accuracy scores, need to fix for multi-class (e.g., if predicted class is present, then true) if column in true label
        # accuracy_df.apply(total_accuracy_prep(x['true_label'], x[column]))
        #total_acc_score = accuracy_score(accuracy_df['true_label'], accuracy_df[column])
        #total_accuracy_scores[column] = [total_acc_score]

    accuracy_scores = pd.DataFrame.from_dict(accuracy_scores, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1']).reset_index().rename(columns={"index":"Model ID"})
    accuracy_scores = accuracy_scores.sort_values(by=['Model ID'])

    daily_accuracy_scores = pd.DataFrame(daily_accuracy).reset_index()
    daily_precision_scores = pd.DataFrame(daily_precision).reset_index()
    daily_recall_scores = pd.DataFrame(daily_recall).reset_index()
    daily_f1_scores = pd.DataFrame(daily_f1).reset_index()

    # Create classification report dataframe


    accuracy_scores['Model Name'] = accuracy_scores.apply(lambda x: get_model_info(x['Model ID'], 'model_name', app_connection),axis=1)
    accuracy_scores['Model Type'] = accuracy_scores.apply(lambda x: get_model_info(x['Model ID'], 'model_type', app_connection),axis=1)
    accuracy_scores['Channels'] = accuracy_scores.apply(lambda x: get_model_info(x['Model ID'], 'channels', app_connection),axis=1)
    accuracy_scores['Active'] = accuracy_scores.apply(lambda x: get_model_info(x['Model ID'], 'active', app_connection),axis=1)

    accuracy_source = ColumnDataSource(data=accuracy_scores)

    columns = [
            TableColumn(field = "Model ID", title="Model ID"),
            TableColumn(field = "Active", title="Active (T/F)"),
            TableColumn(field = "Model Name", title="Model Name"),
            TableColumn(field = "Model Type", title="Model Type"),
            TableColumn(field = "Channels", title="Channels"),
            TableColumn(field = "Accuracy", title="Accuracy"),
            TableColumn(field = "Precision", title="Precision (Weighted)"),
            TableColumn(field = "Recall", title="Recall (Weighted)"),
            TableColumn(field = "F1", title="F1 (Weighted)")
    ]
    prediction_table = DataTable(source=accuracy_source, columns=columns, index_position=None, sizing_mode='stretch_width') #, autosize_mode='fit_viewport'

    # Create data tables for the classification reports
    classification_report_dict = collections.OrderedDict(sorted(classification_report_dict.items()))

    total_report_df = pd.DataFrame()
    for key in classification_report_dict:
        report_df = pd.DataFrame(classification_report_dict[key]).T.reset_index()
        report_df['Model ID'] = key

        total_report_df = pd.concat([total_report_df,report_df], ignore_index=True)

    class_map = {"0":"Class A", "1":"Class B", "2":"Class C", "3":"Class D", "4":"Class E"}

    total_report_df = total_report_df.replace({'index':class_map})


    classification_report_source = ColumnDataSource(data=total_report_df)
    report_columns = [
            TableColumn(field = "index", title="Class/Measure"),
            TableColumn(field = "precision", title="Precision"),
            TableColumn(field = "recall", title="Recall"),
            TableColumn(field = "f1-score", title="F1 Score"),
            TableColumn(field = "support", title="Support"),
            TableColumn(field = "Model ID", title="Model ID")
    ]
    class_report_table = DataTable(source=classification_report_source, columns=report_columns, sizing_mode='stretch_width') #autosize_mode='fit_viewport' 


    # Make multiline daily accuracy plot
    daily_accuracy_plot = figure(width=500, height=300, x_axis_type="datetime", y_range=(0, 1),x_axis_label="Date", y_axis_label="Accuracy Score (Not Cumulative)", title="Daily Accuracy") 

    numlines=len(cols)
    mypalette=[Category10_10[x] for x in range(0,numlines)] #[Spectral11[x] for x in range(0,numlines)]#Spectral11[0:numlines]

    '''
    daily_accuracy_source = ColumnDataSource(dict(
        xs=[daily_accuracy_scores['date']]*numlines,
        ys=[daily_accuracy_scores[column].tolist() for column in cols],
        color=mypalette #[Spectral11[x] for x in range(0,numlines)], #[x for x in mypalette]
    ))

    line_glyph = MultiLine(xs="xs", ys="ys", line_width=2, line_color='color')
    line_renderer = daily_accuracy_plot.add_glyph(daily_accuracy_source, line_glyph)

    # Method using multiple line glyphs
    '''

    daily_accuracy_source = ColumnDataSource(daily_accuracy_scores)


    # Replace cols[i] with the model name
    # Add x axis and y axis labels
    line_renderers=[]
    legend_items=[]
    for i in range(0,len(cols)):
        #glyph = Line(x=daily_accuracy_scores['date'].tolist(), y=daily_accuracy_scores[column[i]].tolist(), line_color=mypalette[i], line_width=6, line_alpha=0.6)
        glyph = Line(x='date', y=cols[i], line_color=mypalette[i], line_width=6, line_alpha=0.6)
        render = daily_accuracy_plot.add_glyph(daily_accuracy_source, glyph)
        #line_renderers.append(render)
        legend_items.append(LegendItem(label=cols[i],renderers=[render]))


    # Add legend to Bokeh plot
    legend=Legend(items=legend_items)
    legend_layout = daily_accuracy_plot.add_layout(legend)


    # Make multiline daily precision plot
    daily_precision_plot = figure(width=500, height=300, x_axis_type="datetime", y_range=(0, 1),x_axis_label="Date", y_axis_label="Precision Score (Not Cumulative)", title="Daily Precision") 

    numlines=len(cols)
    mypalette=[Category10_10[x] for x in range(0,numlines)]

    daily_precision_source = ColumnDataSource(daily_precision_scores)


    # Replace cols[i] with the model name
    # Add x axis and y axis labels
    line_renderers=[]
    legend_items=[]
    for i in range(0,len(cols)):
        #glyph = Line(x=daily_accuracy_scores['date'].tolist(), y=daily_accuracy_scores[column[i]].tolist(), line_color=mypalette[i], line_width=6, line_alpha=0.6)
        glyph = Line(x='date', y=cols[i], line_color=mypalette[i], line_width=6, line_alpha=0.6)
        render = daily_precision_plot.add_glyph(daily_precision_source, glyph)
        #line_renderers.append(render)
        legend_items.append(LegendItem(label=cols[i],renderers=[render]))


    # Add legend to Bokeh plot
    legend=Legend(items=legend_items)
    legend_layout = daily_precision_plot.add_layout(legend)


    # Make multiline daily recall plot
    daily_recall_plot = figure(width=500, height=300, x_axis_type="datetime", y_range=(0, 1),x_axis_label="Date", y_axis_label="Recall Score (Not Cumulative)", title="Daily Recall") 

    numlines=len(cols)
    mypalette=[Category10_10[x] for x in range(0,numlines)]

    daily_recall_source = ColumnDataSource(daily_recall_scores)


    # Replace cols[i] with the model name
    # Add x axis and y axis labels
    line_renderers=[]
    legend_items=[]
    for i in range(0,len(cols)):
        #glyph = Line(x=daily_accuracy_scores['date'].tolist(), y=daily_accuracy_scores[column[i]].tolist(), line_color=mypalette[i], line_width=6, line_alpha=0.6)
        glyph = Line(x='date', y=cols[i], line_color=mypalette[i], line_width=6, line_alpha=0.6)
        render = daily_recall_plot.add_glyph(daily_recall_source, glyph)
        #line_renderers.append(render)
        legend_items.append(LegendItem(label=cols[i],renderers=[render]))


    # Add legend to Bokeh plot
    legend=Legend(items=legend_items)
    legend_layout = daily_recall_plot.add_layout(legend)


    # Make multiline daily F1 Score plot
    daily_f1_plot = figure(width=500, height=300, x_axis_type="datetime", y_range=(0, 1),x_axis_label="Date", y_axis_label="F1 Score (Not Cumulative)", title="Daily F1") 

    numlines=len(cols)
    mypalette=[Category10_10[x] for x in range(0,numlines)]

    daily_f1_source = ColumnDataSource(daily_f1_scores)


    # Replace cols[i] with the model name
    # Add x axis and y axis labels
    line_renderers=[]
    legend_items=[]
    for i in range(0,len(cols)):
        #glyph = Line(x=daily_accuracy_scores['date'].tolist(), y=daily_accuracy_scores[column[i]].tolist(), line_color=mypalette[i], line_width=6, line_alpha=0.6)
        glyph = Line(x='date', y=cols[i], line_color=mypalette[i], line_width=6, line_alpha=0.6)
        render = daily_f1_plot.add_glyph(daily_f1_source, glyph)
        #line_renderers.append(render)
        legend_items.append(LegendItem(label=cols[i],renderers=[render]))


    # Add legend to Bokeh plot
    legend=Legend(items=legend_items)
    legend_layout = daily_f1_plot.add_layout(legend)

    # Create Divs to provide context for Data Tables

    total_metric_div = Div(text="""<h1>Total Metrics for All Models</h1>""",
    sizing_mode='stretch_width', align='center')

    class_report_div = Div(text="""<h1>Classification Reports for All Models</h1>""",
    sizing_mode='stretch_width', align='center')

    daily_metrics_div = Div(text="""<h1>Metrics by Day for All Models</h1>""",
    sizing_mode='stretch_width', align='center')

    explanation_div = Div(text="""<h1><b>Model Performance Metrics</b></h1><br><br> \
                                <h3><b>Multi-Class</b></h3> <h5>Multi-class models are only trained on instances \
                                with one ship present, so their metrics are also only evaluated on instances with one \
                                ship present.</h5><br><br>\
                                \
                                <h3><b>Multi-Label</b></h3> <h5>Multi-label models only make one prediction for each class and \
                                cannot account for multi-instance cases. This is taken into account for computing\
                                metrics for multi-label models, and they are only evaluated on unique occurences of classes.</h5><br><br>""", \
                        sizing_mode='stretch_width', align='center')

    app_connection.close()
    app_engine.dispose()


    curdoc().add_root(layout([[explanation_div],[total_metric_div],[prediction_table],[class_report_div],[class_report_table],[daily_metrics_div],[daily_accuracy_plot, daily_precision_plot],[daily_recall_plot, daily_f1_plot]]))

else:

    explanation_div = Div(text="""<h1><b>Model Performance Metrics</b></h1><br><br> \
                                <h3><b>Multi-Class</b></h3> <h5>Multi-class models are only trained on instances \
                                with one ship present, so their metrics are also only evaluated on instances with one \
                                ship present.</h5><br><br>\
                                \
                                <h3><b>Multi-Label</b></h3> <h5>Multi-label models only make one prediction for each class and \
                                cannot account for multi-instance cases. This is taken into account for computing\
                                metrics for multi-label models, and they are only evaluated on unique occurences of classes.</h5><br><br>""", \
                        sizing_mode='stretch_width', align='center')

    no_metrics_div = Div(text="""<h1>Currently no predictions available to compute metrics. True labels are applied to the previous day's predictions to ensure accurate information from the AIS stream, so the application must have been running for at least 2 days to have populated any metrics data.</h1>""", sizing_mode='stretch_width', align='center')

    curdoc().add_root(layout([[explanation_div], [no_metrics_div]]))

