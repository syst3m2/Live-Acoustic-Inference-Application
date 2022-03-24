from flask_wtf import FlaskForm
from tensorflow.python.ops.gen_math_ops import Select
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, DecimalField, RadioField
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.fields.core import DecimalField
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo
from app.models import User, Models, Ship_Classes
from app import db
from flask_login import current_user
import pandas as pd

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')

class ModelUploadForm(FlaskForm):

    model_type_choices = [('multi_class','multi_class'),('multi_label','multi_label'), ('regression','regression')]
    channel_choices = [(1,1),(4,4)]
    model_input_choices = [('mfcc','mfcc')]
    model_choice_choices = [('resnet1','resnet1'),('dev_bnn_model','dev_bnn_model'),("simple","simple"), \
                            ("resnet2","resnet2"), ("vgg","vgg"), ("inception","inception"), ("mobilenet","mobilenet"), \
                            ("pretrained","pretrained"), ('simple_mc_model','simple_mc_model'), ('simple_bnn_vi_model','simple_bnn_vi_model'), \
                            ('cnn_model','cnn_model'), ('cnn_model_hparam','cnn_model_hparam'), \
                            ('dev_model_hparam','dev_model_hparam')]
    prediction_class_choices = [('classA, classB, classC, classD, classE','classA, classB, classC, classD, classE'), ('ship, not ship','ship, not ship')]

    modelfile = FileField('Upload .h5 model checkpoint file', validators=[FileRequired(),FileAllowed(['h5'], message='Only .h5 model checkpoint files allowed')])
    paramsfile = FileField('Upload corresponding params.txt. Please verify all information is correct, inference will not perform correctly otherwise (e.g. bnn=True)', validators=[FileRequired(),FileAllowed(['txt'], message='Only .txt files allowed')])
    model_name = StringField('Model Name (e.g. Joes Probabilistic Four Channel Model)', validators=[DataRequired()])
    model_type =  SelectField('Model Type', choices=model_type_choices, validators=[DataRequired()])
    channels  = SelectField('Channels (1 or 4 channels)', choices=channel_choices, validators=[DataRequired()])
    model_input = SelectField('Model Input (stft not supported)', choices=model_input_choices, validators=[DataRequired()])
    model_choice = SelectField('Model Choice', choices=model_choice_choices, validators=[DataRequired()])
    prediction_classes = SelectField('Prediction Classes', choices=prediction_class_choices, validators=[DataRequired()])
    training_accuracy_score = DecimalField('Training Accuracy Score (Enter the score the model achieved during training, testing, and validation)', places=4, validators=[DataRequired()])
    submit_upload = SubmitField('Deploy Model for Live Inference')
    
    '''
    def validate_modelfile(self, modelfile):
        if not model_file.endswith('.h5'):
            raise(ValidationError('Only .h5 files are compatible with this application'))
    '''

class ModelDeleteForm(FlaskForm):

    '''
    model_choices =[]
    model_query = "SELECT * FROM models WHERE active=True"
    user_models = pd.read_sql(model_query, db.engine)
    if not user_models.empty:
        for index, row in user_models.iterrows():
            item = str(row['id']) + ' ' + row['model_name']
            model_choices.append((item,item))
    else:
        model_choices.append(('No Models', 'No Models'))

    # Doesn't work
    if current_user.is_authenticated:
        model_query = "SELECT * FROM models WHERE user_id=" + str(current_user.get_id())
        user_models = pd.read_sql(model_query, db.engine)

        if not user_models.empty:
            for index, row in user_models.iterrows():
                model_choices.append((row['model_name'],row['model_name']))
        else:
            model_choices.append(('No Models', 'No Models'))
    else:
        model_choices.append(('No Models', 'No Models'))
    '''

    delete_field = SelectField('Select a model to stop live inference for (Must be linked to your user account)', validators=[DataRequired()])
    submit_delete = SubmitField('Stop Model')

    def __init__(self, *args, **kwargs):
        super(ModelDeleteForm, self).__init__(*args, **kwargs)
        self.delete_field.choices = [(str(h.id) + ' ' + h.model_name, str(h.id) + ' ' + h.model_name) for h in Models.query.filter_by(active=True)]

class ModelActivateForm(FlaskForm):
    '''
    Doesn't work
    model_query = "SELECT * FROM models WHERE user_id=" + str(current_user.get_id())
    user_models = pd.read_sql(model_query, db.engine)
    model_choices =[]

    for index, row in user_models.iterrows():
        model_choices.append((row['model_name'],row['model_name']))
    
    model_query = "SELECT * FROM models WHERE active=False"
    user_models = pd.read_sql(model_query, db.engine)
    model_choices =[]

    for index, row in user_models.iterrows():
        item = str(row['id']) + ' ' + row['model_name']
        #model_choices.append((str(row['id']) + ' ' + row['model_name'], str(row['id']) + ' ' +  row['model_name']))
        model_choices.append((item, item))
    '''
    activate_field = SelectField('Select a model to reactivate (does not delete previous predictions, just halts new inference)', validators=[DataRequired()])
    submit_activate = SubmitField('Reactivate Model')

    def __init__(self, *args, **kwargs):
        super(ModelActivateForm, self).__init__(*args, **kwargs)
        self.activate_field.choices = [(str(h.id) + ' ' + h.model_name, str(h.id) + ' ' + h.model_name) for h in Models.query.filter_by(active=False)]

class ShipClassUpdateForm(FlaskForm):

    desig_field = SelectField('Select a ship designation to update the class for. Some designations may need to remain Unknown due to vague descriptors', validators=[DataRequired()])
    class_select_field = RadioField('Select the Appropriate Ship Class for the Desig', choices=[('Class A','Class A'),('Class B','Class B'), ('Class C','Class C'), ('Class D','Class D'), ('Unknown','Unknown')], validators=[DataRequired()])
    submit_unknown_class = SubmitField('Change Class')

    def __init__(self, *args, **kwargs):
        super(ShipClassUpdateForm, self).__init__(*args, **kwargs)
        self.desig_field.choices = [(h.desig, h.desig) for h in Ship_Classes.query.filter_by(ship_class="Unknown")]

class ChangeShipClassForm(FlaskForm):

    desig_field = SelectField('Select a ship designation to update the class for. Some designations may need to remain Unknown due to vague descriptors', validators=[DataRequired()])
    class_select_field = RadioField('Select the Appropriate Ship Class for the Desig', choices=[('Class A','Class A'),('Class B','Class B'), ('Class C','Class C'), ('Class D','Class D'), ('Unknown','Unknown')], validators=[DataRequired()])
    submit_known_class = SubmitField('Change Class')

    def __init__(self, *args, **kwargs):
        super(ChangeShipClassForm, self).__init__(*args, **kwargs)
        self.desig_field.choices = [(h.desig, h.desig) for h in Ship_Classes.query.filter(Ship_Classes.ship_class!="Unknown")]

    