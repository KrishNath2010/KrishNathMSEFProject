ын
ді
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
ѓ
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.6.02unknown8ЇЦ
є
conv2d_144/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_144/kernel

%conv2d_144/kernel/Read/ReadVariableOpReadVariableOpconv2d_144/kernel*&
_output_shapes
:*
dtype0
v
conv2d_144/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_144/bias
o
#conv2d_144/bias/Read/ReadVariableOpReadVariableOpconv2d_144/bias*
_output_shapes
:*
dtype0
є
conv2d_145/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_145/kernel

%conv2d_145/kernel/Read/ReadVariableOpReadVariableOpconv2d_145/kernel*&
_output_shapes
: *
dtype0
v
conv2d_145/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_145/bias
o
#conv2d_145/bias/Read/ReadVariableOpReadVariableOpconv2d_145/bias*
_output_shapes
: *
dtype0
є
conv2d_146/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_146/kernel

%conv2d_146/kernel/Read/ReadVariableOpReadVariableOpconv2d_146/kernel*&
_output_shapes
: @*
dtype0
v
conv2d_146/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_146/bias
o
#conv2d_146/bias/Read/ReadVariableOpReadVariableOpconv2d_146/bias*
_output_shapes
:@*
dtype0
}
dense_96/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђѓђ* 
shared_namedense_96/kernel
v
#dense_96/kernel/Read/ReadVariableOpReadVariableOpdense_96/kernel*!
_output_shapes
:ђѓђ*
dtype0
s
dense_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_96/bias
l
!dense_96/bias/Read/ReadVariableOpReadVariableOpdense_96/bias*
_output_shapes	
:ђ*
dtype0
{
dense_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ
* 
shared_namedense_97/kernel
t
#dense_97/kernel/Read/ReadVariableOpReadVariableOpdense_97/kernel*
_output_shapes
:	ђ
*
dtype0
r
dense_97/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_97/bias
k
!dense_97/bias/Read/ReadVariableOpReadVariableOpdense_97/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0	
l

Variable_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_1
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0	
l

Variable_2VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_2
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
ћ
Adam/conv2d_144/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_144/kernel/m
Ї
,Adam/conv2d_144/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_144/kernel/m*&
_output_shapes
:*
dtype0
ё
Adam/conv2d_144/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_144/bias/m
}
*Adam/conv2d_144/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_144/bias/m*
_output_shapes
:*
dtype0
ћ
Adam/conv2d_145/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_145/kernel/m
Ї
,Adam/conv2d_145/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_145/kernel/m*&
_output_shapes
: *
dtype0
ё
Adam/conv2d_145/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_145/bias/m
}
*Adam/conv2d_145/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_145/bias/m*
_output_shapes
: *
dtype0
ћ
Adam/conv2d_146/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_146/kernel/m
Ї
,Adam/conv2d_146/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_146/kernel/m*&
_output_shapes
: @*
dtype0
ё
Adam/conv2d_146/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_146/bias/m
}
*Adam/conv2d_146/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_146/bias/m*
_output_shapes
:@*
dtype0
І
Adam/dense_96/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђѓђ*'
shared_nameAdam/dense_96/kernel/m
ё
*Adam/dense_96/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_96/kernel/m*!
_output_shapes
:ђѓђ*
dtype0
Ђ
Adam/dense_96/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_96/bias/m
z
(Adam/dense_96/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_96/bias/m*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_97/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ
*'
shared_nameAdam/dense_97/kernel/m
ѓ
*Adam/dense_97/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_97/kernel/m*
_output_shapes
:	ђ
*
dtype0
ђ
Adam/dense_97/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_97/bias/m
y
(Adam/dense_97/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_97/bias/m*
_output_shapes
:
*
dtype0
ћ
Adam/conv2d_144/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_144/kernel/v
Ї
,Adam/conv2d_144/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_144/kernel/v*&
_output_shapes
:*
dtype0
ё
Adam/conv2d_144/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_144/bias/v
}
*Adam/conv2d_144/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_144/bias/v*
_output_shapes
:*
dtype0
ћ
Adam/conv2d_145/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_145/kernel/v
Ї
,Adam/conv2d_145/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_145/kernel/v*&
_output_shapes
: *
dtype0
ё
Adam/conv2d_145/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_145/bias/v
}
*Adam/conv2d_145/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_145/bias/v*
_output_shapes
: *
dtype0
ћ
Adam/conv2d_146/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_146/kernel/v
Ї
,Adam/conv2d_146/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_146/kernel/v*&
_output_shapes
: @*
dtype0
ё
Adam/conv2d_146/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_146/bias/v
}
*Adam/conv2d_146/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_146/bias/v*
_output_shapes
:@*
dtype0
І
Adam/dense_96/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђѓђ*'
shared_nameAdam/dense_96/kernel/v
ё
*Adam/dense_96/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_96/kernel/v*!
_output_shapes
:ђѓђ*
dtype0
Ђ
Adam/dense_96/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdam/dense_96/bias/v
z
(Adam/dense_96/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_96/bias/v*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_97/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ
*'
shared_nameAdam/dense_97/kernel/v
ѓ
*Adam/dense_97/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_97/kernel/v*
_output_shapes
:	ђ
*
dtype0
ђ
Adam/dense_97/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_97/bias/v
y
(Adam/dense_97/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_97/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
ЯQ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЏQ
valueЉQBјQ BЄQ
Љ
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
y
layer-0
layer-1
layer-2
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
R
$trainable_variables
%	variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
R
.trainable_variables
/	variables
0regularization_losses
1	keras_api
h

2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
R
8trainable_variables
9	variables
:regularization_losses
;	keras_api
R
<trainable_variables
=	variables
>regularization_losses
?	keras_api
R
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
h

Dkernel
Ebias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
h

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
ѕ
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_ratem┬m├(m─)m┼2mк3mКDm╚Em╔Jm╩Km╦v╠v═(v╬)v¤2vл3vЛDvмEvМJvнKvН
F
0
1
(2
)3
24
35
D6
E7
J8
K9
F
0
1
(2
)3
24
35
D6
E7
J8
K9
 
Г
Ulayer_metrics
trainable_variables
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
	variables
regularization_losses
Ymetrics
 
\
Z_rng
[trainable_variables
\	variables
]regularization_losses
^	keras_api
\
__rng
`trainable_variables
a	variables
bregularization_losses
c	keras_api
\
d_rng
etrainable_variables
f	variables
gregularization_losses
h	keras_api
 
 
 
Г
ilayer_metrics
trainable_variables
jlayer_regularization_losses

klayers
lnon_trainable_variables
	variables
regularization_losses
mmetrics
 
 
 
Г
nlayer_metrics
olayer_regularization_losses
trainable_variables

players
qnon_trainable_variables
	variables
regularization_losses
rmetrics
][
VARIABLE_VALUEconv2d_144/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_144/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Г
slayer_metrics
tlayer_regularization_losses
 trainable_variables

ulayers
vnon_trainable_variables
!	variables
"regularization_losses
wmetrics
 
 
 
Г
xlayer_metrics
ylayer_regularization_losses
$trainable_variables

zlayers
{non_trainable_variables
%	variables
&regularization_losses
|metrics
][
VARIABLE_VALUEconv2d_145/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_145/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
»
}layer_metrics
~layer_regularization_losses
*trainable_variables

layers
ђnon_trainable_variables
+	variables
,regularization_losses
Ђmetrics
 
 
 
▓
ѓlayer_metrics
 Ѓlayer_regularization_losses
.trainable_variables
ёlayers
Ёnon_trainable_variables
/	variables
0regularization_losses
єmetrics
][
VARIABLE_VALUEconv2d_146/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_146/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
 
▓
Єlayer_metrics
 ѕlayer_regularization_losses
4trainable_variables
Ѕlayers
іnon_trainable_variables
5	variables
6regularization_losses
Іmetrics
 
 
 
▓
їlayer_metrics
 Їlayer_regularization_losses
8trainable_variables
јlayers
Јnon_trainable_variables
9	variables
:regularization_losses
љmetrics
 
 
 
▓
Љlayer_metrics
 њlayer_regularization_losses
<trainable_variables
Њlayers
ћnon_trainable_variables
=	variables
>regularization_losses
Ћmetrics
 
 
 
▓
ќlayer_metrics
 Ќlayer_regularization_losses
@trainable_variables
ўlayers
Ўnon_trainable_variables
A	variables
Bregularization_losses
џmetrics
[Y
VARIABLE_VALUEdense_96/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_96/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

D0
E1
 
▓
Џlayer_metrics
 юlayer_regularization_losses
Ftrainable_variables
Юlayers
ъnon_trainable_variables
G	variables
Hregularization_losses
Ъmetrics
[Y
VARIABLE_VALUEdense_97/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_97/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
 
▓
аlayer_metrics
 Аlayer_regularization_losses
Ltrainable_variables
бlayers
Бnon_trainable_variables
M	variables
Nregularization_losses
цmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
 

Ц0
д1

Д
_state_var
 
 
 
▓
еlayer_metrics
 Еlayer_regularization_losses
[trainable_variables
фlayers
Фnon_trainable_variables
\	variables
]regularization_losses
гmetrics

Г
_state_var
 
 
 
▓
«layer_metrics
 »layer_regularization_losses
`trainable_variables
░layers
▒non_trainable_variables
a	variables
bregularization_losses
▓metrics

│
_state_var
 
 
 
▓
┤layer_metrics
 хlayer_regularization_losses
etrainable_variables
Хlayers
иnon_trainable_variables
f	variables
gregularization_losses
Иmetrics
 
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

╣total

║count
╗	variables
╝	keras_api
I

йtotal

Йcount
┐
_fn_kwargs
└	variables
┴	keras_api
XV
VARIABLE_VALUEVariable:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
ZX
VARIABLE_VALUE
Variable_1:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
ZX
VARIABLE_VALUE
Variable_2:layer-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

╣0
║1

╗	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

й0
Й1

└	variables
ђ~
VARIABLE_VALUEAdam/conv2d_144/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_144/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/conv2d_145/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_145/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/conv2d_146/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_146/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_96/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_96/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_97/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_97/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/conv2d_144/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_144/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/conv2d_145/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_145/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdam/conv2d_146/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_146/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_96/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_96/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_97/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_97/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
џ
#serving_default_sequential_88_inputPlaceholder*1
_output_shapes
:         ЗЗ*
dtype0*&
shape:         ЗЗ
■
StatefulPartitionedCallStatefulPartitionedCall#serving_default_sequential_88_inputconv2d_144/kernelconv2d_144/biasconv2d_145/kernelconv2d_145/biasconv2d_146/kernelconv2d_146/biasdense_96/kerneldense_96/biasdense_97/kerneldense_97/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_169458
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ц
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_144/kernel/Read/ReadVariableOp#conv2d_144/bias/Read/ReadVariableOp%conv2d_145/kernel/Read/ReadVariableOp#conv2d_145/bias/Read/ReadVariableOp%conv2d_146/kernel/Read/ReadVariableOp#conv2d_146/bias/Read/ReadVariableOp#dense_96/kernel/Read/ReadVariableOp!dense_96/bias/Read/ReadVariableOp#dense_97/kernel/Read/ReadVariableOp!dense_97/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpVariable_2/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv2d_144/kernel/m/Read/ReadVariableOp*Adam/conv2d_144/bias/m/Read/ReadVariableOp,Adam/conv2d_145/kernel/m/Read/ReadVariableOp*Adam/conv2d_145/bias/m/Read/ReadVariableOp,Adam/conv2d_146/kernel/m/Read/ReadVariableOp*Adam/conv2d_146/bias/m/Read/ReadVariableOp*Adam/dense_96/kernel/m/Read/ReadVariableOp(Adam/dense_96/bias/m/Read/ReadVariableOp*Adam/dense_97/kernel/m/Read/ReadVariableOp(Adam/dense_97/bias/m/Read/ReadVariableOp,Adam/conv2d_144/kernel/v/Read/ReadVariableOp*Adam/conv2d_144/bias/v/Read/ReadVariableOp,Adam/conv2d_145/kernel/v/Read/ReadVariableOp*Adam/conv2d_145/bias/v/Read/ReadVariableOp,Adam/conv2d_146/kernel/v/Read/ReadVariableOp*Adam/conv2d_146/bias/v/Read/ReadVariableOp*Adam/dense_96/kernel/v/Read/ReadVariableOp(Adam/dense_96/bias/v/Read/ReadVariableOp*Adam/dense_97/kernel/v/Read/ReadVariableOp(Adam/dense_97/bias/v/Read/ReadVariableOpConst*7
Tin0
.2,				*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *(
f#R!
__inference__traced_save_170914
О
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_144/kernelconv2d_144/biasconv2d_145/kernelconv2d_145/biasconv2d_146/kernelconv2d_146/biasdense_96/kerneldense_96/biasdense_97/kerneldense_97/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateVariable
Variable_1
Variable_2totalcounttotal_1count_1Adam/conv2d_144/kernel/mAdam/conv2d_144/bias/mAdam/conv2d_145/kernel/mAdam/conv2d_145/bias/mAdam/conv2d_146/kernel/mAdam/conv2d_146/bias/mAdam/dense_96/kernel/mAdam/dense_96/bias/mAdam/dense_97/kernel/mAdam/dense_97/bias/mAdam/conv2d_144/kernel/vAdam/conv2d_144/bias/vAdam/conv2d_145/kernel/vAdam/conv2d_145/bias/vAdam/conv2d_146/kernel/vAdam/conv2d_146/bias/vAdam/dense_96/kernel/vAdam/dense_96/bias/vAdam/dense_97/kernel/vAdam/dense_97/bias/v*6
Tin/
-2+*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_171050┤С
ц
f
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_168571

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
ф4
љ
I__inference_sequential_89_layer_call_and_return_conditional_losses_169110

inputs+
conv2d_144_169019:
conv2d_144_169021:+
conv2d_145_169037: 
conv2d_145_169039: +
conv2d_146_169055: @
conv2d_146_169057:@$
dense_96_169088:ђѓђ
dense_96_169090:	ђ"
dense_97_169104:	ђ

dense_97_169106:

identityѕб"conv2d_144/StatefulPartitionedCallб"conv2d_145/StatefulPartitionedCallб"conv2d_146/StatefulPartitionedCallб dense_96/StatefulPartitionedCallб dense_97/StatefulPartitionedCallь
sequential_88/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1685742
sequential_88/PartitionedCallі
rescaling_51/PartitionedCallPartitionedCall&sequential_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_1690052
rescaling_51/PartitionedCallК
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall%rescaling_51/PartitionedCall:output:0conv2d_144_169019conv2d_144_169021*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_144_layer_call_and_return_conditional_losses_1690182$
"conv2d_144/StatefulPartitionedCallъ
!max_pooling2d_144/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЩЩ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_1689592#
!max_pooling2d_144/PartitionedCall╠
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_144/PartitionedCall:output:0conv2d_145_169037conv2d_145_169039*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЩЩ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_145_layer_call_and_return_conditional_losses_1690362$
"conv2d_145/StatefulPartitionedCallю
!max_pooling2d_145/PartitionedCallPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }} * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_1689712#
!max_pooling2d_145/PartitionedCall╩
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_145/PartitionedCall:output:0conv2d_146_169055conv2d_146_169057*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_146_layer_call_and_return_conditional_losses_1690542$
"conv2d_146/StatefulPartitionedCallю
!max_pooling2d_146/PartitionedCallPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_1689832#
!max_pooling2d_146/PartitionedCallє
dropout_37/PartitionedCallPartitionedCall*max_pooling2d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1690662
dropout_37/PartitionedCallщ
flatten_48/PartitionedCallPartitionedCall#dropout_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђѓ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_48_layer_call_and_return_conditional_losses_1690742
flatten_48/PartitionedCall▓
 dense_96/StatefulPartitionedCallStatefulPartitionedCall#flatten_48/PartitionedCall:output:0dense_96_169088dense_96_169090*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_1690872"
 dense_96/StatefulPartitionedCallи
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_169104dense_97_169106*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_1691032"
 dense_97/StatefulPartitionedCallё
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

IdentityЃ
NoOpNoOp#^conv2d_144/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЗЗ: : : : : : : : : : 2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
ч
џ
)__inference_dense_96_layer_call_fn_170304

inputs
unknown:ђѓђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_1690872
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђѓ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         ђѓ
 
_user_specified_nameinputs
Ж
b
F__inference_flatten_48_layer_call_and_return_conditional_losses_170279

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     ┴ 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђѓ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђѓ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         >>@:W S
/
_output_shapes
:         >>@
 
_user_specified_nameinputs
џ
s
I__inference_sequential_88_layer_call_and_return_conditional_losses_168940
random_flip_41_input
identity■
random_flip_41/PartitionedCallPartitionedCallrandom_flip_41_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1685592 
random_flip_41/PartitionedCallЮ
"random_rotation_40/PartitionedCallPartitionedCall'random_flip_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1685652$
"random_rotation_40/PartitionedCallЋ
random_zoom_40/PartitionedCallPartitionedCall+random_rotation_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1685712 
random_zoom_40/PartitionedCallЁ
IdentityIdentity'random_zoom_40/PartitionedCall:output:0*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:g c
1
_output_shapes
:         ЗЗ
.
_user_specified_namerandom_flip_41_input
ф

Ш
D__inference_dense_97_layer_call_and_return_conditional_losses_170314

inputs1
matmul_readvariableop_resource:	ђ
-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
э┤
у
"__inference__traced_restore_171050
file_prefix<
"assignvariableop_conv2d_144_kernel:0
"assignvariableop_1_conv2d_144_bias:>
$assignvariableop_2_conv2d_145_kernel: 0
"assignvariableop_3_conv2d_145_bias: >
$assignvariableop_4_conv2d_146_kernel: @0
"assignvariableop_5_conv2d_146_bias:@7
"assignvariableop_6_dense_96_kernel:ђѓђ/
 assignvariableop_7_dense_96_bias:	ђ5
"assignvariableop_8_dense_97_kernel:	ђ
.
 assignvariableop_9_dense_97_bias:
'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: *
assignvariableop_15_variable:	,
assignvariableop_16_variable_1:	,
assignvariableop_17_variable_2:	#
assignvariableop_18_total: #
assignvariableop_19_count: %
assignvariableop_20_total_1: %
assignvariableop_21_count_1: F
,assignvariableop_22_adam_conv2d_144_kernel_m:8
*assignvariableop_23_adam_conv2d_144_bias_m:F
,assignvariableop_24_adam_conv2d_145_kernel_m: 8
*assignvariableop_25_adam_conv2d_145_bias_m: F
,assignvariableop_26_adam_conv2d_146_kernel_m: @8
*assignvariableop_27_adam_conv2d_146_bias_m:@?
*assignvariableop_28_adam_dense_96_kernel_m:ђѓђ7
(assignvariableop_29_adam_dense_96_bias_m:	ђ=
*assignvariableop_30_adam_dense_97_kernel_m:	ђ
6
(assignvariableop_31_adam_dense_97_bias_m:
F
,assignvariableop_32_adam_conv2d_144_kernel_v:8
*assignvariableop_33_adam_conv2d_144_bias_v:F
,assignvariableop_34_adam_conv2d_145_kernel_v: 8
*assignvariableop_35_adam_conv2d_145_bias_v: F
,assignvariableop_36_adam_conv2d_146_kernel_v: @8
*assignvariableop_37_adam_conv2d_146_bias_v:@?
*assignvariableop_38_adam_dense_96_kernel_v:ђѓђ7
(assignvariableop_39_adam_dense_96_bias_v:	ђ=
*assignvariableop_40_adam_dense_97_kernel_v:	ђ
6
(assignvariableop_41_adam_dense_97_bias_v:

identity_43ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9║
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*к
value╝B╣+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesС
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЁ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*┬
_output_shapes»
г:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+				2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityА
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_144_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Д
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_144_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Е
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_145_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Д
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_145_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Е
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv2d_146_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Д
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_146_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Д
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_96_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ц
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_96_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Д
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_97_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ц
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_97_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10Ц
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Д
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Д
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13д
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_15ц
AssignVariableOp_15AssignVariableOpassignvariableop_15_variableIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16д
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_17д
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18А
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19А
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Б
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Б
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22┤
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_conv2d_144_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23▓
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_144_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24┤
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_conv2d_145_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▓
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_145_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26┤
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_conv2d_146_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27▓
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_146_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28▓
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_96_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29░
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_96_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▓
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_97_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31░
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense_97_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32┤
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_conv2d_144_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33▓
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_144_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34┤
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_conv2d_145_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35▓
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_145_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36┤
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_conv2d_146_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37▓
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_146_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▓
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_96_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39░
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense_96_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40▓
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_97_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41░
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_97_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_419
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЩ
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_42f
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_43Р
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_43Identity_43:output:0*i
_input_shapesX
V: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
­
K
/__inference_random_flip_41_layer_call_fn_170491

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1685592
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
ф

Ш
D__inference_dense_97_layer_call_and_return_conditional_losses_169103

inputs1
matmul_readvariableop_resource:	ђ
-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
№
f
J__inference_random_flip_41_layer_call_and_return_conditional_losses_168353

inputs
identity}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ь
e
F__inference_dropout_37_layer_call_and_return_conditional_losses_169179

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *GГќ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         >>@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         >>@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *г>2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         >>@2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         >>@2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         >>@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         >>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         >>@:W S
/
_output_shapes
:         >>@
 
_user_specified_nameinputs
ЗЃ
├
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170738

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityѕбstateful_uniform/RngReadAndSkipD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1v
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/shape/1А
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *═╠ї?2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/ConstЎ
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform/Prodt
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/Cast/xі
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1┘
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkipќ
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stackџ
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1џ
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2╬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_sliceЎ
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcastџ
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stackъ
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1ъ
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2к
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1Ъ
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1а
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg╝
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2+
)stateful_uniform/StatelessRandomUniformV2њ
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub│
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         2
stateful_uniform/mulў
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:         2
stateful_uniform\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЎ
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concate
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
zoom_matrix/Shapeї
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
zoom_matrix/strided_slice/stackљ
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_1љ
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_2ф
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zoom_matrix/strided_slicek
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
zoom_matrix/sub/yr
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/subs
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv/yІ
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truedivЏ
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_1/stackЪ
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_1/stack_1Ъ
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_1/stack_2ы
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_1o
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
zoom_matrix/sub_1/xБ
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/sub_1І
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         2
zoom_matrix/mulo
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
zoom_matrix/sub_2/yv
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/sub_2w
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv_1/yЊ
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv_1Џ
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_2/stackЪ
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_2/stack_1Ъ
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_2/stack_2ы
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_2o
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
zoom_matrix/sub_3/xБ
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/sub_3Љ
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         2
zoom_matrix/mul_1Џ
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_3/stackЪ
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_3/stack_1Ъ
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_3/stack_2ы
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_3z
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros/packed/1│
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros/packedw
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros/ConstЦ
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros~
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/packed/1╣
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_1/packed{
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_1/ConstГ
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros_1Џ
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_4/stackЪ
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_4/stack_1Ъ
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_4/stack_2ы
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_4~
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_2/packed/1╣
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_2/packed{
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_2/ConstГ
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros_2t
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/concat/axisр
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
zoom_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shapeѕ
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stackї
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1ї
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2і
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceq
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
transform/fill_value┼
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:         ЗЗ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3ъ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ЗЗ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
ц
f
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170636

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
ы

Њ
$__inference_signature_wrapper_169458
sequential_88_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:ђѓђ
	unknown_6:	ђ
	unknown_7:	ђ

	unknown_8:

identityѕбStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallsequential_88_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_1683452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЗЗ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:         ЗЗ
-
_user_specified_namesequential_88_input
Г
i
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_168971

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Њ
d
F__inference_dropout_37_layer_call_and_return_conditional_losses_170251

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         >>@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         >>@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         >>@:W S
/
_output_shapes
:         >>@
 
_user_specified_nameinputs
├
░
.__inference_sequential_88_layer_call_fn_170173

inputs
unknown:	
	unknown_0:	
	unknown_1:	
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1689132
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ЗЗ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
њ
щ
D__inference_dense_96_layer_call_and_return_conditional_losses_169087

inputs3
matmul_readvariableop_resource:ђѓђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpљ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђѓђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђѓ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         ђѓ
 
_user_specified_nameinputs
Э
 
F__inference_conv2d_145_layer_call_and_return_conditional_losses_169036

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЩЩ *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЩЩ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЩЩ 2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЩЩ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЩЩ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЩЩ
 
_user_specified_nameinputs
Сf
ђ
J__inference_random_flip_41_layer_call_and_return_conditional_losses_168888

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityѕб(stateful_uniform_full_int/RngReadAndSkipбOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgбVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterї
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shapeї
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Constй
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prodє
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/xЦ
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1є
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkipе
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stackг
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1г
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2ё
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice┤
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcastг
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack░
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1░
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2Ч
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1║
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1ђ
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg«
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2
stateful_uniform_full_intb

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2

zeros_likeЂ
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:2
stack{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ѕ
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceН
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:         ЗЗ25
3stateless_random_flip_left_right/control_dependency╝
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/ShapeХ
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4stateless_random_flip_left_right/strided_slice/stack║
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_1║
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2е
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.stateless_random_flip_left_right/strided_sliceы
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shape├
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/min├
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2?
=stateless_random_flip_left_right/stateless_random_uniform/maxі
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterг
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg╩
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:         2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2Х
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subМ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         2?
=stateless_random_flip_left_right/stateless_random_uniform/mulИ
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         2;
9stateless_random_flip_left_right/stateless_random_uniformд
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1д
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2д
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3ђ
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shapeЉ
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         2*
(stateless_random_flip_left_right/Reshapeк
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         2(
&stateless_random_flip_left_right/Roundг
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axisЎ
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         ЗЗ2,
*stateless_random_flip_left_right/ReverseV2­
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ЗЗ2&
$stateless_random_flip_left_right/mulЋ
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2(
&stateless_random_flip_left_right/sub/xЖ
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         2&
$stateless_random_flip_left_right/subч
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         ЗЗ2(
&stateless_random_flip_left_right/mul_1у
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ЗЗ2&
$stateless_random_flip_left_right/addЇ
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityц
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ЗЗ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2б
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2░
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
ЗЃ
├
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_168687

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityѕбstateful_uniform/RngReadAndSkipD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1v
stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/shape/1А
stateful_uniform/shapePackstrided_slice:output:0!stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *═╠ї?2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/ConstЎ
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform/Prodt
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/Cast/xі
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1┘
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkipќ
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stackџ
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1џ
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2╬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_sliceЎ
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcastџ
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stackъ
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1ъ
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2к
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1Ъ
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1а
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg╝
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2+
)stateful_uniform/StatelessRandomUniformV2њ
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub│
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         2
stateful_uniform/mulў
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:         2
stateful_uniform\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisЎ
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concate
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
zoom_matrix/Shapeї
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
zoom_matrix/strided_slice/stackљ
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_1љ
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_2ф
zoom_matrix/strided_sliceStridedSlicezoom_matrix/Shape:output:0(zoom_matrix/strided_slice/stack:output:0*zoom_matrix/strided_slice/stack_1:output:0*zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
zoom_matrix/strided_slicek
zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
zoom_matrix/sub/yr
zoom_matrix/subSub
Cast_1:y:0zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/subs
zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv/yІ
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truedivЏ
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_1/stackЪ
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_1/stack_1Ъ
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_1/stack_2ы
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_1o
zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
zoom_matrix/sub_1/xБ
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/sub_1І
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         2
zoom_matrix/mulo
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
zoom_matrix/sub_2/yv
zoom_matrix/sub_2SubCast:y:0zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/sub_2w
zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
zoom_matrix/truediv_1/yЊ
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv_1Џ
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_2/stackЪ
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_2/stack_1Ъ
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_2/stack_2ы
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_2o
zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
zoom_matrix/sub_3/xБ
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/sub_3Љ
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         2
zoom_matrix/mul_1Џ
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_3/stackЪ
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_3/stack_1Ъ
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_3/stack_2ы
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_3z
zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros/packed/1│
zoom_matrix/zeros/packedPack"zoom_matrix/strided_slice:output:0#zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros/packedw
zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros/ConstЦ
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros~
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/packed/1╣
zoom_matrix/zeros_1/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_1/packed{
zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_1/ConstГ
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros_1Џ
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_4/stackЪ
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_4/stack_1Ъ
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_4/stack_2ы
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2
zoom_matrix/strided_slice_4~
zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_2/packed/1╣
zoom_matrix/zeros_2/packedPack"zoom_matrix/strided_slice:output:0%zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2
zoom_matrix/zeros_2/packed{
zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zoom_matrix/zeros_2/ConstГ
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         2
zoom_matrix/zeros_2t
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/concat/axisр
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
zoom_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shapeѕ
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stackї
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1ї
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2і
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceq
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
transform/fill_value┼
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:         ЗЗ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3ъ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ЗЗ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
█
N
2__inference_max_pooling2d_144_layer_call_fn_168965

inputs
identityЬ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_1689592
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Э
 
F__inference_conv2d_144_layer_call_and_return_conditional_losses_169018

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЗЗ*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЗЗ2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЗЗ2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЗЗ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Я
G
+__inference_dropout_37_layer_call_fn_170268

inputs
identity╠
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1690662
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         >>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         >>@:W S
/
_output_shapes
:         >>@
 
_user_specified_nameinputs
њ
щ
D__inference_dense_96_layer_call_and_return_conditional_losses_170295

inputs3
matmul_readvariableop_resource:ђѓђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpљ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:ђѓђ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:         ђѓ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:         ђѓ
 
_user_specified_nameinputs
Ж
b
F__inference_flatten_48_layer_call_and_return_conditional_losses_169074

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     ┴ 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         ђѓ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         ђѓ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         >>@:W S
/
_output_shapes
:         >>@
 
_user_specified_nameinputs
д
а
+__inference_conv2d_144_layer_call_fn_170206

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_144_layer_call_and_return_conditional_losses_1690182
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЗЗ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
З
Ќ
)__inference_dense_97_layer_call_fn_170323

inputs
unknown:	ђ

	unknown_0:

identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_1691032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
­
K
/__inference_random_zoom_40_layer_call_fn_170743

inputs
identityм
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1685712
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
ф
d
H__inference_rescaling_51_layer_call_and_return_conditional_losses_170181

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ђђђ;2
Cast/xY
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2

Cast_1/xf
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:         ЗЗ2
mulk
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:         ЗЗ2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Ќ
У
I__inference_sequential_88_layer_call_and_return_conditional_losses_168953
random_flip_41_input#
random_flip_41_168943:	'
random_rotation_40_168946:	#
random_zoom_40_168949:	
identityѕб&random_flip_41/StatefulPartitionedCallб*random_rotation_40/StatefulPartitionedCallб&random_zoom_40/StatefulPartitionedCall«
&random_flip_41/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_41_inputrandom_flip_41_168943*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1688882(
&random_flip_41/StatefulPartitionedCall┘
*random_rotation_40/StatefulPartitionedCallStatefulPartitionedCall/random_flip_41/StatefulPartitionedCall:output:0random_rotation_40_168946*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1688182,
*random_rotation_40/StatefulPartitionedCall═
&random_zoom_40/StatefulPartitionedCallStatefulPartitionedCall3random_rotation_40/StatefulPartitionedCall:output:0random_zoom_40_168949*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1686872(
&random_zoom_40/StatefulPartitionedCallћ
IdentityIdentity/random_zoom_40/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identity═
NoOpNoOp'^random_flip_41/StatefulPartitionedCall+^random_rotation_40/StatefulPartitionedCall'^random_zoom_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ЗЗ: : : 2P
&random_flip_41/StatefulPartitionedCall&random_flip_41/StatefulPartitionedCall2X
*random_rotation_40/StatefulPartitionedCall*random_rotation_40/StatefulPartitionedCall2P
&random_zoom_40/StatefulPartitionedCall&random_zoom_40/StatefulPartitionedCall:g c
1
_output_shapes
:         ЗЗ
.
_user_specified_namerandom_flip_41_input
█
N
2__inference_max_pooling2d_146_layer_call_fn_168989

inputs
identityЬ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_1689832
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
В
I
-__inference_rescaling_51_layer_call_fn_170186

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_1690052
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
СX
Ў
__inference__traced_save_170914
file_prefix0
,savev2_conv2d_144_kernel_read_readvariableop.
*savev2_conv2d_144_bias_read_readvariableop0
,savev2_conv2d_145_kernel_read_readvariableop.
*savev2_conv2d_145_bias_read_readvariableop0
,savev2_conv2d_146_kernel_read_readvariableop.
*savev2_conv2d_146_bias_read_readvariableop.
*savev2_dense_96_kernel_read_readvariableop,
(savev2_dense_96_bias_read_readvariableop.
*savev2_dense_97_kernel_read_readvariableop,
(savev2_dense_97_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop'
#savev2_variable_read_readvariableop	)
%savev2_variable_1_read_readvariableop	)
%savev2_variable_2_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop7
3savev2_adam_conv2d_144_kernel_m_read_readvariableop5
1savev2_adam_conv2d_144_bias_m_read_readvariableop7
3savev2_adam_conv2d_145_kernel_m_read_readvariableop5
1savev2_adam_conv2d_145_bias_m_read_readvariableop7
3savev2_adam_conv2d_146_kernel_m_read_readvariableop5
1savev2_adam_conv2d_146_bias_m_read_readvariableop5
1savev2_adam_dense_96_kernel_m_read_readvariableop3
/savev2_adam_dense_96_bias_m_read_readvariableop5
1savev2_adam_dense_97_kernel_m_read_readvariableop3
/savev2_adam_dense_97_bias_m_read_readvariableop7
3savev2_adam_conv2d_144_kernel_v_read_readvariableop5
1savev2_adam_conv2d_144_bias_v_read_readvariableop7
3savev2_adam_conv2d_145_kernel_v_read_readvariableop5
1savev2_adam_conv2d_145_bias_v_read_readvariableop7
3savev2_adam_conv2d_146_kernel_v_read_readvariableop5
1savev2_adam_conv2d_146_bias_v_read_readvariableop5
1savev2_adam_dense_96_kernel_v_read_readvariableop3
/savev2_adam_dense_96_bias_v_read_readvariableop5
1savev2_adam_dense_97_kernel_v_read_readvariableop3
/savev2_adam_dense_97_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename┤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*к
value╝B╣+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesя
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesж
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_144_kernel_read_readvariableop*savev2_conv2d_144_bias_read_readvariableop,savev2_conv2d_145_kernel_read_readvariableop*savev2_conv2d_145_bias_read_readvariableop,savev2_conv2d_146_kernel_read_readvariableop*savev2_conv2d_146_bias_read_readvariableop*savev2_dense_96_kernel_read_readvariableop(savev2_dense_96_bias_read_readvariableop*savev2_dense_97_kernel_read_readvariableop(savev2_dense_97_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_144_kernel_m_read_readvariableop1savev2_adam_conv2d_144_bias_m_read_readvariableop3savev2_adam_conv2d_145_kernel_m_read_readvariableop1savev2_adam_conv2d_145_bias_m_read_readvariableop3savev2_adam_conv2d_146_kernel_m_read_readvariableop1savev2_adam_conv2d_146_bias_m_read_readvariableop1savev2_adam_dense_96_kernel_m_read_readvariableop/savev2_adam_dense_96_bias_m_read_readvariableop1savev2_adam_dense_97_kernel_m_read_readvariableop/savev2_adam_dense_97_bias_m_read_readvariableop3savev2_adam_conv2d_144_kernel_v_read_readvariableop1savev2_adam_conv2d_144_bias_v_read_readvariableop3savev2_adam_conv2d_145_kernel_v_read_readvariableop1savev2_adam_conv2d_145_bias_v_read_readvariableop3savev2_adam_conv2d_146_kernel_v_read_readvariableop1savev2_adam_conv2d_146_bias_v_read_readvariableop1savev2_adam_dense_96_kernel_v_read_readvariableop/savev2_adam_dense_96_bias_v_read_readvariableop1savev2_adam_dense_97_kernel_v_read_readvariableop/savev2_adam_dense_97_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+				2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*ё
_input_shapesЫ
№: ::: : : @:@:ђѓђ:ђ:	ђ
:
: : : : : :::: : : : ::: : : @:@:ђѓђ:ђ:	ђ
:
::: : : @:@:ђѓђ:ђ:	ђ
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:'#
!
_output_shapes
:ђѓђ:!

_output_shapes	
:ђ:%	!

_output_shapes
:	ђ
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:'#
!
_output_shapes
:ђѓђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ
:  

_output_shapes
:
:,!(
&
_output_shapes
:: "

_output_shapes
::,#(
&
_output_shapes
: : $

_output_shapes
: :,%(
&
_output_shapes
: @: &

_output_shapes
:@:''#
!
_output_shapes
:ђѓђ:!(

_output_shapes	
:ђ:%)!

_output_shapes
:	ђ
: *

_output_shapes
:
:+

_output_shapes
: 
ъ
а
+__inference_conv2d_146_layer_call_fn_170246

inputs!
unknown: @
	unknown_0:@
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_146_layer_call_and_return_conditional_losses_1690542
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         }}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         }} : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         }} 
 
_user_specified_nameinputs
н
G
+__inference_flatten_48_layer_call_fn_170284

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђѓ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_48_layer_call_and_return_conditional_losses_1690742
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         ђѓ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         >>@:W S
/
_output_shapes
:         >>@
 
_user_specified_nameinputs
ц
f
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170416

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
├
d
+__inference_dropout_37_layer_call_fn_170273

inputs
identityѕбStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1691792
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         >>@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         >>@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         >>@
 
_user_specified_nameinputs
Н8
╔
I__inference_sequential_89_layer_call_and_return_conditional_losses_169287

inputs"
sequential_88_169248:	"
sequential_88_169250:	"
sequential_88_169252:	+
conv2d_144_169256:
conv2d_144_169258:+
conv2d_145_169262: 
conv2d_145_169264: +
conv2d_146_169268: @
conv2d_146_169270:@$
dense_96_169276:ђѓђ
dense_96_169278:	ђ"
dense_97_169281:	ђ

dense_97_169283:

identityѕб"conv2d_144/StatefulPartitionedCallб"conv2d_145/StatefulPartitionedCallб"conv2d_146/StatefulPartitionedCallб dense_96/StatefulPartitionedCallб dense_97/StatefulPartitionedCallб"dropout_37/StatefulPartitionedCallб%sequential_88/StatefulPartitionedCall╩
%sequential_88/StatefulPartitionedCallStatefulPartitionedCallinputssequential_88_169248sequential_88_169250sequential_88_169252*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1689132'
%sequential_88/StatefulPartitionedCallњ
rescaling_51/PartitionedCallPartitionedCall.sequential_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_1690052
rescaling_51/PartitionedCallК
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall%rescaling_51/PartitionedCall:output:0conv2d_144_169256conv2d_144_169258*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_144_layer_call_and_return_conditional_losses_1690182$
"conv2d_144/StatefulPartitionedCallъ
!max_pooling2d_144/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЩЩ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_1689592#
!max_pooling2d_144/PartitionedCall╠
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_144/PartitionedCall:output:0conv2d_145_169262conv2d_145_169264*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЩЩ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_145_layer_call_and_return_conditional_losses_1690362$
"conv2d_145/StatefulPartitionedCallю
!max_pooling2d_145/PartitionedCallPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }} * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_1689712#
!max_pooling2d_145/PartitionedCall╩
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_145/PartitionedCall:output:0conv2d_146_169268conv2d_146_169270*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_146_layer_call_and_return_conditional_losses_1690542$
"conv2d_146/StatefulPartitionedCallю
!max_pooling2d_146/PartitionedCallPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_1689832#
!max_pooling2d_146/PartitionedCallъ
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1691792$
"dropout_37/StatefulPartitionedCallЂ
flatten_48/PartitionedCallPartitionedCall+dropout_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђѓ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_48_layer_call_and_return_conditional_losses_1690742
flatten_48/PartitionedCall▓
 dense_96/StatefulPartitionedCallStatefulPartitionedCall#flatten_48/PartitionedCall:output:0dense_96_169276dense_96_169278*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_1690872"
 dense_96/StatefulPartitionedCallи
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_169281dense_97_169283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_1691032"
 dense_97/StatefulPartitionedCallё
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityл
NoOpNoOp#^conv2d_144/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall&^sequential_88/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         ЗЗ: : : : : : : : : : : : : 2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2N
%sequential_88/StatefulPartitionedCall%sequential_88/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
пЊ
Т
I__inference_sequential_89_layer_call_and_return_conditional_losses_169827

inputs\
Nsequential_88_random_flip_41_stateful_uniform_full_int_rngreadandskip_resource:	W
Isequential_88_random_rotation_40_stateful_uniform_rngreadandskip_resource:	S
Esequential_88_random_zoom_40_stateful_uniform_rngreadandskip_resource:	C
)conv2d_144_conv2d_readvariableop_resource:8
*conv2d_144_biasadd_readvariableop_resource:C
)conv2d_145_conv2d_readvariableop_resource: 8
*conv2d_145_biasadd_readvariableop_resource: C
)conv2d_146_conv2d_readvariableop_resource: @8
*conv2d_146_biasadd_readvariableop_resource:@<
'dense_96_matmul_readvariableop_resource:ђѓђ7
(dense_96_biasadd_readvariableop_resource:	ђ:
'dense_97_matmul_readvariableop_resource:	ђ
6
(dense_97_biasadd_readvariableop_resource:

identityѕб!conv2d_144/BiasAdd/ReadVariableOpб conv2d_144/Conv2D/ReadVariableOpб!conv2d_145/BiasAdd/ReadVariableOpб conv2d_145/Conv2D/ReadVariableOpб!conv2d_146/BiasAdd/ReadVariableOpб conv2d_146/Conv2D/ReadVariableOpбdense_96/BiasAdd/ReadVariableOpбdense_96/MatMul/ReadVariableOpбdense_97/BiasAdd/ReadVariableOpбdense_97/MatMul/ReadVariableOpбEsequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkipбlsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgбssequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterб@sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkipб<sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkipк
<sequential_88/random_flip_41/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_88/random_flip_41/stateful_uniform_full_int/shapeк
<sequential_88/random_flip_41/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_88/random_flip_41/stateful_uniform_full_int/Const▒
;sequential_88/random_flip_41/stateful_uniform_full_int/ProdProdEsequential_88/random_flip_41/stateful_uniform_full_int/shape:output:0Esequential_88/random_flip_41/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2=
;sequential_88/random_flip_41/stateful_uniform_full_int/Prod└
=sequential_88/random_flip_41/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_88/random_flip_41/stateful_uniform_full_int/Cast/xЧ
=sequential_88/random_flip_41/stateful_uniform_full_int/Cast_1CastDsequential_88/random_flip_41/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2?
=sequential_88/random_flip_41/stateful_uniform_full_int/Cast_1Ќ
Esequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipNsequential_88_random_flip_41_stateful_uniform_full_int_rngreadandskip_resourceFsequential_88/random_flip_41/stateful_uniform_full_int/Cast/x:output:0Asequential_88/random_flip_41/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2G
Esequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkipР
Jsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2L
Jsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stackТ
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_1Т
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_2▓
Dsequential_88/random_flip_41/stateful_uniform_full_int/strided_sliceStridedSliceMsequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkip:value:0Ssequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack:output:0Usequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_1:output:0Usequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2F
Dsequential_88/random_flip_41/stateful_uniform_full_int/strided_sliceІ
>sequential_88/random_flip_41/stateful_uniform_full_int/BitcastBitcastMsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02@
>sequential_88/random_flip_41/stateful_uniform_full_int/BitcastТ
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stackЖ
Nsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2P
Nsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1Ж
Nsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2P
Nsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2ф
Fsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1StridedSliceMsequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkip:value:0Usequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack:output:0Wsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Wsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2H
Fsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1Љ
@sequential_88/random_flip_41/stateful_uniform_full_int/Bitcast_1BitcastOsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02B
@sequential_88/random_flip_41/stateful_uniform_full_int/Bitcast_1║
:sequential_88/random_flip_41/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_88/random_flip_41/stateful_uniform_full_int/alg▄
6sequential_88/random_flip_41/stateful_uniform_full_intStatelessRandomUniformFullIntV2Esequential_88/random_flip_41/stateful_uniform_full_int/shape:output:0Isequential_88/random_flip_41/stateful_uniform_full_int/Bitcast_1:output:0Gsequential_88/random_flip_41/stateful_uniform_full_int/Bitcast:output:0Csequential_88/random_flip_41/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	28
6sequential_88/random_flip_41/stateful_uniform_full_intю
'sequential_88/random_flip_41/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2)
'sequential_88/random_flip_41/zeros_likeш
"sequential_88/random_flip_41/stackPack?sequential_88/random_flip_41/stateful_uniform_full_int:output:00sequential_88/random_flip_41/zeros_like:output:0*
N*
T0	*
_output_shapes

:2$
"sequential_88/random_flip_41/stackх
0sequential_88/random_flip_41/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0sequential_88/random_flip_41/strided_slice/stack╣
2sequential_88/random_flip_41/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2sequential_88/random_flip_41/strided_slice/stack_1╣
2sequential_88/random_flip_41/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2sequential_88/random_flip_41/strided_slice/stack_2Х
*sequential_88/random_flip_41/strided_sliceStridedSlice+sequential_88/random_flip_41/stack:output:09sequential_88/random_flip_41/strided_slice/stack:output:0;sequential_88/random_flip_41/strided_slice/stack_1:output:0;sequential_88/random_flip_41/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2,
*sequential_88/random_flip_41/strided_sliceЈ
Psequential_88/random_flip_41/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:         ЗЗ2R
Psequential_88/random_flip_41/stateless_random_flip_left_right/control_dependencyЊ
Csequential_88/random_flip_41/stateless_random_flip_left_right/ShapeShapeYsequential_88/random_flip_41/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2E
Csequential_88/random_flip_41/stateless_random_flip_left_right/Shape­
Qsequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
Qsequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stackЗ
Ssequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_1З
Ssequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_2о
Ksequential_88/random_flip_41/stateless_random_flip_left_right/strided_sliceStridedSliceLsequential_88/random_flip_41/stateless_random_flip_left_right/Shape:output:0Zsequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack:output:0\sequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_1:output:0\sequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2M
Ksequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice╚
\sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shapePackTsequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2^
\sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shape§
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2\
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/min§
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2\
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/maxр
ssequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter3sequential_88/random_flip_41/strided_slice:output:0* 
_output_shapes
::2u
ssequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterЃ
lsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgt^sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2n
lsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgЭ
osequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2esequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0ysequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0}sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0rsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:         2q
osequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2ф
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/subSubcsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/max:output:0csequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2\
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/subК
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mulMulxsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0^sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         2\
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mulг
Vsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniformAddV2^sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0csequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         2X
Vsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniformЯ
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2O
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/1Я
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2O
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/2Я
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2O
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/3«
Ksequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shapePackTsequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice:output:0Vsequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/1:output:0Vsequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/2:output:0Vsequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2M
Ksequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shapeЁ
Esequential_88/random_flip_41/stateless_random_flip_left_right/ReshapeReshapeZsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform:z:0Tsequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         2G
Esequential_88/random_flip_41/stateless_random_flip_left_right/ReshapeЮ
Csequential_88/random_flip_41/stateless_random_flip_left_right/RoundRoundNsequential_88/random_flip_41/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         2E
Csequential_88/random_flip_41/stateless_random_flip_left_right/RoundТ
Lsequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2/axisЇ
Gsequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2	ReverseV2Ysequential_88/random_flip_41/stateless_random_flip_left_right/control_dependency:output:0Usequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         ЗЗ2I
Gsequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2С
Asequential_88/random_flip_41/stateless_random_flip_left_right/mulMulGsequential_88/random_flip_41/stateless_random_flip_left_right/Round:y:0Psequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ЗЗ2C
Asequential_88/random_flip_41/stateless_random_flip_left_right/mul¤
Csequential_88/random_flip_41/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2E
Csequential_88/random_flip_41/stateless_random_flip_left_right/sub/xя
Asequential_88/random_flip_41/stateless_random_flip_left_right/subSubLsequential_88/random_flip_41/stateless_random_flip_left_right/sub/x:output:0Gsequential_88/random_flip_41/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         2C
Asequential_88/random_flip_41/stateless_random_flip_left_right/sub№
Csequential_88/random_flip_41/stateless_random_flip_left_right/mul_1MulEsequential_88/random_flip_41/stateless_random_flip_left_right/sub:z:0Ysequential_88/random_flip_41/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         ЗЗ2E
Csequential_88/random_flip_41/stateless_random_flip_left_right/mul_1█
Asequential_88/random_flip_41/stateless_random_flip_left_right/addAddV2Esequential_88/random_flip_41/stateless_random_flip_left_right/mul:z:0Gsequential_88/random_flip_41/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ЗЗ2C
Asequential_88/random_flip_41/stateless_random_flip_left_right/add┼
&sequential_88/random_rotation_40/ShapeShapeEsequential_88/random_flip_41/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2(
&sequential_88/random_rotation_40/ShapeХ
4sequential_88/random_rotation_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_88/random_rotation_40/strided_slice/stack║
6sequential_88/random_rotation_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_88/random_rotation_40/strided_slice/stack_1║
6sequential_88/random_rotation_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_88/random_rotation_40/strided_slice/stack_2е
.sequential_88/random_rotation_40/strided_sliceStridedSlice/sequential_88/random_rotation_40/Shape:output:0=sequential_88/random_rotation_40/strided_slice/stack:output:0?sequential_88/random_rotation_40/strided_slice/stack_1:output:0?sequential_88/random_rotation_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_88/random_rotation_40/strided_slice║
6sequential_88/random_rotation_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6sequential_88/random_rotation_40/strided_slice_1/stackЙ
8sequential_88/random_rotation_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_88/random_rotation_40/strided_slice_1/stack_1Й
8sequential_88/random_rotation_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_88/random_rotation_40/strided_slice_1/stack_2▓
0sequential_88/random_rotation_40/strided_slice_1StridedSlice/sequential_88/random_rotation_40/Shape:output:0?sequential_88/random_rotation_40/strided_slice_1/stack:output:0Asequential_88/random_rotation_40/strided_slice_1/stack_1:output:0Asequential_88/random_rotation_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_88/random_rotation_40/strided_slice_1┴
%sequential_88/random_rotation_40/CastCast9sequential_88/random_rotation_40/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%sequential_88/random_rotation_40/Cast║
6sequential_88/random_rotation_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6sequential_88/random_rotation_40/strided_slice_2/stackЙ
8sequential_88/random_rotation_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_88/random_rotation_40/strided_slice_2/stack_1Й
8sequential_88/random_rotation_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_88/random_rotation_40/strided_slice_2/stack_2▓
0sequential_88/random_rotation_40/strided_slice_2StridedSlice/sequential_88/random_rotation_40/Shape:output:0?sequential_88/random_rotation_40/strided_slice_2/stack:output:0Asequential_88/random_rotation_40/strided_slice_2/stack_1:output:0Asequential_88/random_rotation_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_88/random_rotation_40/strided_slice_2┼
'sequential_88/random_rotation_40/Cast_1Cast9sequential_88/random_rotation_40/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'sequential_88/random_rotation_40/Cast_1р
7sequential_88/random_rotation_40/stateful_uniform/shapePack7sequential_88/random_rotation_40/strided_slice:output:0*
N*
T0*
_output_shapes
:29
7sequential_88/random_rotation_40/stateful_uniform/shape│
5sequential_88/random_rotation_40/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5sequential_88/random_rotation_40/stateful_uniform/min│
5sequential_88/random_rotation_40/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5sequential_88/random_rotation_40/stateful_uniform/max╝
7sequential_88/random_rotation_40/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_88/random_rotation_40/stateful_uniform/ConstЮ
6sequential_88/random_rotation_40/stateful_uniform/ProdProd@sequential_88/random_rotation_40/stateful_uniform/shape:output:0@sequential_88/random_rotation_40/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/stateful_uniform/ProdХ
8sequential_88/random_rotation_40/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_88/random_rotation_40/stateful_uniform/Cast/xь
8sequential_88/random_rotation_40/stateful_uniform/Cast_1Cast?sequential_88/random_rotation_40/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2:
8sequential_88/random_rotation_40/stateful_uniform/Cast_1■
@sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkipRngReadAndSkipIsequential_88_random_rotation_40_stateful_uniform_rngreadandskip_resourceAsequential_88/random_rotation_40/stateful_uniform/Cast/x:output:0<sequential_88/random_rotation_40/stateful_uniform/Cast_1:y:0*
_output_shapes
:2B
@sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkipп
Esequential_88/random_rotation_40/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_88/random_rotation_40/stateful_uniform/strided_slice/stack▄
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_1▄
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_2ћ
?sequential_88/random_rotation_40/stateful_uniform/strided_sliceStridedSliceHsequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip:value:0Nsequential_88/random_rotation_40/stateful_uniform/strided_slice/stack:output:0Psequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_1:output:0Psequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2A
?sequential_88/random_rotation_40/stateful_uniform/strided_sliceЧ
9sequential_88/random_rotation_40/stateful_uniform/BitcastBitcastHsequential_88/random_rotation_40/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02;
9sequential_88/random_rotation_40/stateful_uniform/Bitcast▄
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stackЯ
Isequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_1Я
Isequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_2ї
Asequential_88/random_rotation_40/stateful_uniform/strided_slice_1StridedSliceHsequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip:value:0Psequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack:output:0Rsequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_1:output:0Rsequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2C
Asequential_88/random_rotation_40/stateful_uniform/strided_slice_1ѓ
;sequential_88/random_rotation_40/stateful_uniform/Bitcast_1BitcastJsequential_88/random_rotation_40/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02=
;sequential_88/random_rotation_40/stateful_uniform/Bitcast_1Р
Nsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2P
Nsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2/alg■
Jsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2@sequential_88/random_rotation_40/stateful_uniform/shape:output:0Dsequential_88/random_rotation_40/stateful_uniform/Bitcast_1:output:0Bsequential_88/random_rotation_40/stateful_uniform/Bitcast:output:0Wsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         2L
Jsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2ќ
5sequential_88/random_rotation_40/stateful_uniform/subSub>sequential_88/random_rotation_40/stateful_uniform/max:output:0>sequential_88/random_rotation_40/stateful_uniform/min:output:0*
T0*
_output_shapes
: 27
5sequential_88/random_rotation_40/stateful_uniform/sub│
5sequential_88/random_rotation_40/stateful_uniform/mulMulSsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2:output:09sequential_88/random_rotation_40/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         27
5sequential_88/random_rotation_40/stateful_uniform/mulў
1sequential_88/random_rotation_40/stateful_uniformAddV29sequential_88/random_rotation_40/stateful_uniform/mul:z:0>sequential_88/random_rotation_40/stateful_uniform/min:output:0*
T0*#
_output_shapes
:         23
1sequential_88/random_rotation_40/stateful_uniformх
6sequential_88/random_rotation_40/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?28
6sequential_88/random_rotation_40/rotation_matrix/sub/yѓ
4sequential_88/random_rotation_40/rotation_matrix/subSub+sequential_88/random_rotation_40/Cast_1:y:0?sequential_88/random_rotation_40/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 26
4sequential_88/random_rotation_40/rotation_matrix/subп
4sequential_88/random_rotation_40/rotation_matrix/CosCos5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         26
4sequential_88/random_rotation_40/rotation_matrix/Cos╣
8sequential_88/random_rotation_40/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2:
8sequential_88/random_rotation_40/rotation_matrix/sub_1/yѕ
6sequential_88/random_rotation_40/rotation_matrix/sub_1Sub+sequential_88/random_rotation_40/Cast_1:y:0Asequential_88/random_rotation_40/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/rotation_matrix/sub_1Ќ
4sequential_88/random_rotation_40/rotation_matrix/mulMul8sequential_88/random_rotation_40/rotation_matrix/Cos:y:0:sequential_88/random_rotation_40/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         26
4sequential_88/random_rotation_40/rotation_matrix/mulп
4sequential_88/random_rotation_40/rotation_matrix/SinSin5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         26
4sequential_88/random_rotation_40/rotation_matrix/Sin╣
8sequential_88/random_rotation_40/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2:
8sequential_88/random_rotation_40/rotation_matrix/sub_2/yє
6sequential_88/random_rotation_40/rotation_matrix/sub_2Sub)sequential_88/random_rotation_40/Cast:y:0Asequential_88/random_rotation_40/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/rotation_matrix/sub_2Џ
6sequential_88/random_rotation_40/rotation_matrix/mul_1Mul8sequential_88/random_rotation_40/rotation_matrix/Sin:y:0:sequential_88/random_rotation_40/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/mul_1Џ
6sequential_88/random_rotation_40/rotation_matrix/sub_3Sub8sequential_88/random_rotation_40/rotation_matrix/mul:z:0:sequential_88/random_rotation_40/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/sub_3Џ
6sequential_88/random_rotation_40/rotation_matrix/sub_4Sub8sequential_88/random_rotation_40/rotation_matrix/sub:z:0:sequential_88/random_rotation_40/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/sub_4й
:sequential_88/random_rotation_40/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:sequential_88/random_rotation_40/rotation_matrix/truediv/y«
8sequential_88/random_rotation_40/rotation_matrix/truedivRealDiv:sequential_88/random_rotation_40/rotation_matrix/sub_4:z:0Csequential_88/random_rotation_40/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         2:
8sequential_88/random_rotation_40/rotation_matrix/truediv╣
8sequential_88/random_rotation_40/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2:
8sequential_88/random_rotation_40/rotation_matrix/sub_5/yє
6sequential_88/random_rotation_40/rotation_matrix/sub_5Sub)sequential_88/random_rotation_40/Cast:y:0Asequential_88/random_rotation_40/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/rotation_matrix/sub_5▄
6sequential_88/random_rotation_40/rotation_matrix/Sin_1Sin5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/Sin_1╣
8sequential_88/random_rotation_40/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2:
8sequential_88/random_rotation_40/rotation_matrix/sub_6/yѕ
6sequential_88/random_rotation_40/rotation_matrix/sub_6Sub+sequential_88/random_rotation_40/Cast_1:y:0Asequential_88/random_rotation_40/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/rotation_matrix/sub_6Ю
6sequential_88/random_rotation_40/rotation_matrix/mul_2Mul:sequential_88/random_rotation_40/rotation_matrix/Sin_1:y:0:sequential_88/random_rotation_40/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/mul_2▄
6sequential_88/random_rotation_40/rotation_matrix/Cos_1Cos5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/Cos_1╣
8sequential_88/random_rotation_40/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2:
8sequential_88/random_rotation_40/rotation_matrix/sub_7/yє
6sequential_88/random_rotation_40/rotation_matrix/sub_7Sub)sequential_88/random_rotation_40/Cast:y:0Asequential_88/random_rotation_40/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/rotation_matrix/sub_7Ю
6sequential_88/random_rotation_40/rotation_matrix/mul_3Mul:sequential_88/random_rotation_40/rotation_matrix/Cos_1:y:0:sequential_88/random_rotation_40/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/mul_3Џ
4sequential_88/random_rotation_40/rotation_matrix/addAddV2:sequential_88/random_rotation_40/rotation_matrix/mul_2:z:0:sequential_88/random_rotation_40/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         26
4sequential_88/random_rotation_40/rotation_matrix/addЏ
6sequential_88/random_rotation_40/rotation_matrix/sub_8Sub:sequential_88/random_rotation_40/rotation_matrix/sub_5:z:08sequential_88/random_rotation_40/rotation_matrix/add:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/sub_8┴
<sequential_88/random_rotation_40/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2>
<sequential_88/random_rotation_40/rotation_matrix/truediv_1/y┤
:sequential_88/random_rotation_40/rotation_matrix/truediv_1RealDiv:sequential_88/random_rotation_40/rotation_matrix/sub_8:z:0Esequential_88/random_rotation_40/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         2<
:sequential_88/random_rotation_40/rotation_matrix/truediv_1Н
6sequential_88/random_rotation_40/rotation_matrix/ShapeShape5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*
_output_shapes
:28
6sequential_88/random_rotation_40/rotation_matrix/Shapeо
Dsequential_88/random_rotation_40/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_88/random_rotation_40/rotation_matrix/strided_slice/stack┌
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_1┌
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_2ѕ
>sequential_88/random_rotation_40/rotation_matrix/strided_sliceStridedSlice?sequential_88/random_rotation_40/rotation_matrix/Shape:output:0Msequential_88/random_rotation_40/rotation_matrix/strided_slice/stack:output:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_1:output:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>sequential_88/random_rotation_40/rotation_matrix/strided_slice▄
6sequential_88/random_rotation_40/rotation_matrix/Cos_2Cos5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/Cos_2р
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stackт
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_1т
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_2й
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_1StridedSlice:sequential_88/random_rotation_40/rotation_matrix/Cos_2:y:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_1▄
6sequential_88/random_rotation_40/rotation_matrix/Sin_2Sin5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/Sin_2р
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stackт
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_1т
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_2й
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_2StridedSlice:sequential_88/random_rotation_40/rotation_matrix/Sin_2:y:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_2­
4sequential_88/random_rotation_40/rotation_matrix/NegNegIsequential_88/random_rotation_40/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         26
4sequential_88/random_rotation_40/rotation_matrix/Negр
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stackт
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_1т
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_2┐
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_3StridedSlice<sequential_88/random_rotation_40/rotation_matrix/truediv:z:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_3▄
6sequential_88/random_rotation_40/rotation_matrix/Sin_3Sin5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/Sin_3р
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stackт
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_1т
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_2й
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_4StridedSlice:sequential_88/random_rotation_40/rotation_matrix/Sin_3:y:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_4▄
6sequential_88/random_rotation_40/rotation_matrix/Cos_3Cos5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/Cos_3р
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stackт
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_1т
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_2й
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_5StridedSlice:sequential_88/random_rotation_40/rotation_matrix/Cos_3:y:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_5р
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stackт
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_1т
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_2┴
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_6StridedSlice>sequential_88/random_rotation_40/rotation_matrix/truediv_1:z:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_6─
?sequential_88/random_rotation_40/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_88/random_rotation_40/rotation_matrix/zeros/packed/1К
=sequential_88/random_rotation_40/rotation_matrix/zeros/packedPackGsequential_88/random_rotation_40/rotation_matrix/strided_slice:output:0Hsequential_88/random_rotation_40/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2?
=sequential_88/random_rotation_40/rotation_matrix/zeros/packed┴
<sequential_88/random_rotation_40/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2>
<sequential_88/random_rotation_40/rotation_matrix/zeros/Const╣
6sequential_88/random_rotation_40/rotation_matrix/zerosFillFsequential_88/random_rotation_40/rotation_matrix/zeros/packed:output:0Esequential_88/random_rotation_40/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         28
6sequential_88/random_rotation_40/rotation_matrix/zerosЙ
<sequential_88/random_rotation_40/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2>
<sequential_88/random_rotation_40/rotation_matrix/concat/axisЫ
7sequential_88/random_rotation_40/rotation_matrix/concatConcatV2Isequential_88/random_rotation_40/rotation_matrix/strided_slice_1:output:08sequential_88/random_rotation_40/rotation_matrix/Neg:y:0Isequential_88/random_rotation_40/rotation_matrix/strided_slice_3:output:0Isequential_88/random_rotation_40/rotation_matrix/strided_slice_4:output:0Isequential_88/random_rotation_40/rotation_matrix/strided_slice_5:output:0Isequential_88/random_rotation_40/rotation_matrix/strided_slice_6:output:0?sequential_88/random_rotation_40/rotation_matrix/zeros:output:0Esequential_88/random_rotation_40/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         29
7sequential_88/random_rotation_40/rotation_matrix/concat┘
0sequential_88/random_rotation_40/transform/ShapeShapeEsequential_88/random_flip_41/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:22
0sequential_88/random_rotation_40/transform/Shape╩
>sequential_88/random_rotation_40/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_88/random_rotation_40/transform/strided_slice/stack╬
@sequential_88/random_rotation_40/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_88/random_rotation_40/transform/strided_slice/stack_1╬
@sequential_88/random_rotation_40/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_88/random_rotation_40/transform/strided_slice/stack_2л
8sequential_88/random_rotation_40/transform/strided_sliceStridedSlice9sequential_88/random_rotation_40/transform/Shape:output:0Gsequential_88/random_rotation_40/transform/strided_slice/stack:output:0Isequential_88/random_rotation_40/transform/strided_slice/stack_1:output:0Isequential_88/random_rotation_40/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2:
8sequential_88/random_rotation_40/transform/strided_slice│
5sequential_88/random_rotation_40/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5sequential_88/random_rotation_40/transform/fill_valueГ
Esequential_88/random_rotation_40/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Esequential_88/random_flip_41/stateless_random_flip_left_right/add:z:0@sequential_88/random_rotation_40/rotation_matrix/concat:output:0Asequential_88/random_rotation_40/transform/strided_slice:output:0>sequential_88/random_rotation_40/transform/fill_value:output:0*1
_output_shapes
:         ЗЗ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2G
Esequential_88/random_rotation_40/transform/ImageProjectiveTransformV3м
"sequential_88/random_zoom_40/ShapeShapeZsequential_88/random_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2$
"sequential_88/random_zoom_40/Shape«
0sequential_88/random_zoom_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_88/random_zoom_40/strided_slice/stack▓
2sequential_88/random_zoom_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_88/random_zoom_40/strided_slice/stack_1▓
2sequential_88/random_zoom_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_88/random_zoom_40/strided_slice/stack_2љ
*sequential_88/random_zoom_40/strided_sliceStridedSlice+sequential_88/random_zoom_40/Shape:output:09sequential_88/random_zoom_40/strided_slice/stack:output:0;sequential_88/random_zoom_40/strided_slice/stack_1:output:0;sequential_88/random_zoom_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_88/random_zoom_40/strided_slice▓
2sequential_88/random_zoom_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2sequential_88/random_zoom_40/strided_slice_1/stackХ
4sequential_88/random_zoom_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_88/random_zoom_40/strided_slice_1/stack_1Х
4sequential_88/random_zoom_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_88/random_zoom_40/strided_slice_1/stack_2џ
,sequential_88/random_zoom_40/strided_slice_1StridedSlice+sequential_88/random_zoom_40/Shape:output:0;sequential_88/random_zoom_40/strided_slice_1/stack:output:0=sequential_88/random_zoom_40/strided_slice_1/stack_1:output:0=sequential_88/random_zoom_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_88/random_zoom_40/strided_slice_1х
!sequential_88/random_zoom_40/CastCast5sequential_88/random_zoom_40/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!sequential_88/random_zoom_40/Cast▓
2sequential_88/random_zoom_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2sequential_88/random_zoom_40/strided_slice_2/stackХ
4sequential_88/random_zoom_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_88/random_zoom_40/strided_slice_2/stack_1Х
4sequential_88/random_zoom_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_88/random_zoom_40/strided_slice_2/stack_2џ
,sequential_88/random_zoom_40/strided_slice_2StridedSlice+sequential_88/random_zoom_40/Shape:output:0;sequential_88/random_zoom_40/strided_slice_2/stack:output:0=sequential_88/random_zoom_40/strided_slice_2/stack_1:output:0=sequential_88/random_zoom_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_88/random_zoom_40/strided_slice_2╣
#sequential_88/random_zoom_40/Cast_1Cast5sequential_88/random_zoom_40/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#sequential_88/random_zoom_40/Cast_1░
5sequential_88/random_zoom_40/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5sequential_88/random_zoom_40/stateful_uniform/shape/1Ћ
3sequential_88/random_zoom_40/stateful_uniform/shapePack3sequential_88/random_zoom_40/strided_slice:output:0>sequential_88/random_zoom_40/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:25
3sequential_88/random_zoom_40/stateful_uniform/shapeФ
1sequential_88/random_zoom_40/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?23
1sequential_88/random_zoom_40/stateful_uniform/minФ
1sequential_88/random_zoom_40/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *═╠ї?23
1sequential_88/random_zoom_40/stateful_uniform/max┤
3sequential_88/random_zoom_40/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_88/random_zoom_40/stateful_uniform/ConstЇ
2sequential_88/random_zoom_40/stateful_uniform/ProdProd<sequential_88/random_zoom_40/stateful_uniform/shape:output:0<sequential_88/random_zoom_40/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 24
2sequential_88/random_zoom_40/stateful_uniform/Prod«
4sequential_88/random_zoom_40/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_88/random_zoom_40/stateful_uniform/Cast/xр
4sequential_88/random_zoom_40/stateful_uniform/Cast_1Cast;sequential_88/random_zoom_40/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 26
4sequential_88/random_zoom_40/stateful_uniform/Cast_1Ж
<sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkipRngReadAndSkipEsequential_88_random_zoom_40_stateful_uniform_rngreadandskip_resource=sequential_88/random_zoom_40/stateful_uniform/Cast/x:output:08sequential_88/random_zoom_40/stateful_uniform/Cast_1:y:0*
_output_shapes
:2>
<sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkipл
Asequential_88/random_zoom_40/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_88/random_zoom_40/stateful_uniform/strided_slice/stackн
Csequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_1н
Csequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_2Ч
;sequential_88/random_zoom_40/stateful_uniform/strided_sliceStridedSliceDsequential_88/random_zoom_40/stateful_uniform/RngReadAndSkip:value:0Jsequential_88/random_zoom_40/stateful_uniform/strided_slice/stack:output:0Lsequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_1:output:0Lsequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2=
;sequential_88/random_zoom_40/stateful_uniform/strided_slice­
5sequential_88/random_zoom_40/stateful_uniform/BitcastBitcastDsequential_88/random_zoom_40/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type027
5sequential_88/random_zoom_40/stateful_uniform/Bitcastн
Csequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stackп
Esequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_1п
Esequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_2З
=sequential_88/random_zoom_40/stateful_uniform/strided_slice_1StridedSliceDsequential_88/random_zoom_40/stateful_uniform/RngReadAndSkip:value:0Lsequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack:output:0Nsequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_1:output:0Nsequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2?
=sequential_88/random_zoom_40/stateful_uniform/strided_slice_1Ш
7sequential_88/random_zoom_40/stateful_uniform/Bitcast_1BitcastFsequential_88/random_zoom_40/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type029
7sequential_88/random_zoom_40/stateful_uniform/Bitcast_1┌
Jsequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2L
Jsequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2/algЖ
Fsequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2<sequential_88/random_zoom_40/stateful_uniform/shape:output:0@sequential_88/random_zoom_40/stateful_uniform/Bitcast_1:output:0>sequential_88/random_zoom_40/stateful_uniform/Bitcast:output:0Ssequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2H
Fsequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2є
1sequential_88/random_zoom_40/stateful_uniform/subSub:sequential_88/random_zoom_40/stateful_uniform/max:output:0:sequential_88/random_zoom_40/stateful_uniform/min:output:0*
T0*
_output_shapes
: 23
1sequential_88/random_zoom_40/stateful_uniform/subД
1sequential_88/random_zoom_40/stateful_uniform/mulMulOsequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2:output:05sequential_88/random_zoom_40/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         23
1sequential_88/random_zoom_40/stateful_uniform/mulї
-sequential_88/random_zoom_40/stateful_uniformAddV25sequential_88/random_zoom_40/stateful_uniform/mul:z:0:sequential_88/random_zoom_40/stateful_uniform/min:output:0*
T0*'
_output_shapes
:         2/
-sequential_88/random_zoom_40/stateful_uniformќ
(sequential_88/random_zoom_40/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_88/random_zoom_40/concat/axisф
#sequential_88/random_zoom_40/concatConcatV21sequential_88/random_zoom_40/stateful_uniform:z:01sequential_88/random_zoom_40/stateful_uniform:z:01sequential_88/random_zoom_40/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2%
#sequential_88/random_zoom_40/concat╝
.sequential_88/random_zoom_40/zoom_matrix/ShapeShape,sequential_88/random_zoom_40/concat:output:0*
T0*
_output_shapes
:20
.sequential_88/random_zoom_40/zoom_matrix/Shapeк
<sequential_88/random_zoom_40/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_88/random_zoom_40/zoom_matrix/strided_slice/stack╩
>sequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_1╩
>sequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_2п
6sequential_88/random_zoom_40/zoom_matrix/strided_sliceStridedSlice7sequential_88/random_zoom_40/zoom_matrix/Shape:output:0Esequential_88/random_zoom_40/zoom_matrix/strided_slice/stack:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_1:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6sequential_88/random_zoom_40/zoom_matrix/strided_sliceЦ
.sequential_88/random_zoom_40/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?20
.sequential_88/random_zoom_40/zoom_matrix/sub/yТ
,sequential_88/random_zoom_40/zoom_matrix/subSub'sequential_88/random_zoom_40/Cast_1:y:07sequential_88/random_zoom_40/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2.
,sequential_88/random_zoom_40/zoom_matrix/subГ
2sequential_88/random_zoom_40/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @24
2sequential_88/random_zoom_40/zoom_matrix/truediv/y 
0sequential_88/random_zoom_40/zoom_matrix/truedivRealDiv0sequential_88/random_zoom_40/zoom_matrix/sub:z:0;sequential_88/random_zoom_40/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 22
0sequential_88/random_zoom_40/zoom_matrix/truedivН
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack┘
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_1┘
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_2Ъ
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_1StridedSlice,sequential_88/random_zoom_40/concat:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_1:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2:
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_1Е
0sequential_88/random_zoom_40/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?22
0sequential_88/random_zoom_40/zoom_matrix/sub_1/xЌ
.sequential_88/random_zoom_40/zoom_matrix/sub_1Sub9sequential_88/random_zoom_40/zoom_matrix/sub_1/x:output:0Asequential_88/random_zoom_40/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         20
.sequential_88/random_zoom_40/zoom_matrix/sub_1 
,sequential_88/random_zoom_40/zoom_matrix/mulMul4sequential_88/random_zoom_40/zoom_matrix/truediv:z:02sequential_88/random_zoom_40/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         2.
,sequential_88/random_zoom_40/zoom_matrix/mulЕ
0sequential_88/random_zoom_40/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?22
0sequential_88/random_zoom_40/zoom_matrix/sub_2/yЖ
.sequential_88/random_zoom_40/zoom_matrix/sub_2Sub%sequential_88/random_zoom_40/Cast:y:09sequential_88/random_zoom_40/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 20
.sequential_88/random_zoom_40/zoom_matrix/sub_2▒
4sequential_88/random_zoom_40/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @26
4sequential_88/random_zoom_40/zoom_matrix/truediv_1/yЄ
2sequential_88/random_zoom_40/zoom_matrix/truediv_1RealDiv2sequential_88/random_zoom_40/zoom_matrix/sub_2:z:0=sequential_88/random_zoom_40/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 24
2sequential_88/random_zoom_40/zoom_matrix/truediv_1Н
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack┘
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_1┘
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_2Ъ
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_2StridedSlice,sequential_88/random_zoom_40/concat:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_1:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2:
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_2Е
0sequential_88/random_zoom_40/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?22
0sequential_88/random_zoom_40/zoom_matrix/sub_3/xЌ
.sequential_88/random_zoom_40/zoom_matrix/sub_3Sub9sequential_88/random_zoom_40/zoom_matrix/sub_3/x:output:0Asequential_88/random_zoom_40/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         20
.sequential_88/random_zoom_40/zoom_matrix/sub_3Ё
.sequential_88/random_zoom_40/zoom_matrix/mul_1Mul6sequential_88/random_zoom_40/zoom_matrix/truediv_1:z:02sequential_88/random_zoom_40/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         20
.sequential_88/random_zoom_40/zoom_matrix/mul_1Н
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack┘
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_1┘
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_2Ъ
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_3StridedSlice,sequential_88/random_zoom_40/concat:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_1:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2:
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_3┤
7sequential_88/random_zoom_40/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :29
7sequential_88/random_zoom_40/zoom_matrix/zeros/packed/1Д
5sequential_88/random_zoom_40/zoom_matrix/zeros/packedPack?sequential_88/random_zoom_40/zoom_matrix/strided_slice:output:0@sequential_88/random_zoom_40/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:27
5sequential_88/random_zoom_40/zoom_matrix/zeros/packed▒
4sequential_88/random_zoom_40/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4sequential_88/random_zoom_40/zoom_matrix/zeros/ConstЎ
.sequential_88/random_zoom_40/zoom_matrix/zerosFill>sequential_88/random_zoom_40/zoom_matrix/zeros/packed:output:0=sequential_88/random_zoom_40/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         20
.sequential_88/random_zoom_40/zoom_matrix/zerosИ
9sequential_88/random_zoom_40/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2;
9sequential_88/random_zoom_40/zoom_matrix/zeros_1/packed/1Г
7sequential_88/random_zoom_40/zoom_matrix/zeros_1/packedPack?sequential_88/random_zoom_40/zoom_matrix/strided_slice:output:0Bsequential_88/random_zoom_40/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:29
7sequential_88/random_zoom_40/zoom_matrix/zeros_1/packedх
6sequential_88/random_zoom_40/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6sequential_88/random_zoom_40/zoom_matrix/zeros_1/ConstА
0sequential_88/random_zoom_40/zoom_matrix/zeros_1Fill@sequential_88/random_zoom_40/zoom_matrix/zeros_1/packed:output:0?sequential_88/random_zoom_40/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         22
0sequential_88/random_zoom_40/zoom_matrix/zeros_1Н
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack┘
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_1┘
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_2Ъ
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_4StridedSlice,sequential_88/random_zoom_40/concat:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_1:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2:
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_4И
9sequential_88/random_zoom_40/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2;
9sequential_88/random_zoom_40/zoom_matrix/zeros_2/packed/1Г
7sequential_88/random_zoom_40/zoom_matrix/zeros_2/packedPack?sequential_88/random_zoom_40/zoom_matrix/strided_slice:output:0Bsequential_88/random_zoom_40/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:29
7sequential_88/random_zoom_40/zoom_matrix/zeros_2/packedх
6sequential_88/random_zoom_40/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6sequential_88/random_zoom_40/zoom_matrix/zeros_2/ConstА
0sequential_88/random_zoom_40/zoom_matrix/zeros_2Fill@sequential_88/random_zoom_40/zoom_matrix/zeros_2/packed:output:0?sequential_88/random_zoom_40/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         22
0sequential_88/random_zoom_40/zoom_matrix/zeros_2«
4sequential_88/random_zoom_40/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_88/random_zoom_40/zoom_matrix/concat/axisЃ
/sequential_88/random_zoom_40/zoom_matrix/concatConcatV2Asequential_88/random_zoom_40/zoom_matrix/strided_slice_3:output:07sequential_88/random_zoom_40/zoom_matrix/zeros:output:00sequential_88/random_zoom_40/zoom_matrix/mul:z:09sequential_88/random_zoom_40/zoom_matrix/zeros_1:output:0Asequential_88/random_zoom_40/zoom_matrix/strided_slice_4:output:02sequential_88/random_zoom_40/zoom_matrix/mul_1:z:09sequential_88/random_zoom_40/zoom_matrix/zeros_2:output:0=sequential_88/random_zoom_40/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         21
/sequential_88/random_zoom_40/zoom_matrix/concatТ
,sequential_88/random_zoom_40/transform/ShapeShapeZsequential_88/random_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2.
,sequential_88/random_zoom_40/transform/Shape┬
:sequential_88/random_zoom_40/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_88/random_zoom_40/transform/strided_slice/stackк
<sequential_88/random_zoom_40/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_88/random_zoom_40/transform/strided_slice/stack_1к
<sequential_88/random_zoom_40/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_88/random_zoom_40/transform/strided_slice/stack_2И
4sequential_88/random_zoom_40/transform/strided_sliceStridedSlice5sequential_88/random_zoom_40/transform/Shape:output:0Csequential_88/random_zoom_40/transform/strided_slice/stack:output:0Esequential_88/random_zoom_40/transform/strided_slice/stack_1:output:0Esequential_88/random_zoom_40/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:26
4sequential_88/random_zoom_40/transform/strided_sliceФ
1sequential_88/random_zoom_40/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1sequential_88/random_zoom_40/transform/fill_valueф
Asequential_88/random_zoom_40/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Zsequential_88/random_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:08sequential_88/random_zoom_40/zoom_matrix/concat:output:0=sequential_88/random_zoom_40/transform/strided_slice:output:0:sequential_88/random_zoom_40/transform/fill_value:output:0*1
_output_shapes
:         ЗЗ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2C
Asequential_88/random_zoom_40/transform/ImageProjectiveTransformV3o
rescaling_51/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ђђђ;2
rescaling_51/Cast/xs
rescaling_51/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_51/Cast_1/xП
rescaling_51/mulMulVsequential_88/random_zoom_40/transform/ImageProjectiveTransformV3:transformed_images:0rescaling_51/Cast/x:output:0*
T0*1
_output_shapes
:         ЗЗ2
rescaling_51/mulЪ
rescaling_51/addAddV2rescaling_51/mul:z:0rescaling_51/Cast_1/x:output:0*
T0*1
_output_shapes
:         ЗЗ2
rescaling_51/addХ
 conv2d_144/Conv2D/ReadVariableOpReadVariableOp)conv2d_144_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_144/Conv2D/ReadVariableOpн
conv2d_144/Conv2DConv2Drescaling_51/add:z:0(conv2d_144/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЗЗ*
paddingSAME*
strides
2
conv2d_144/Conv2DГ
!conv2d_144/BiasAdd/ReadVariableOpReadVariableOp*conv2d_144_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_144/BiasAdd/ReadVariableOpХ
conv2d_144/BiasAddBiasAddconv2d_144/Conv2D:output:0)conv2d_144/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЗЗ2
conv2d_144/BiasAddЃ
conv2d_144/ReluReluconv2d_144/BiasAdd:output:0*
T0*1
_output_shapes
:         ЗЗ2
conv2d_144/Relu¤
max_pooling2d_144/MaxPoolMaxPoolconv2d_144/Relu:activations:0*1
_output_shapes
:         ЩЩ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_144/MaxPoolХ
 conv2d_145/Conv2D/ReadVariableOpReadVariableOp)conv2d_145_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 conv2d_145/Conv2D/ReadVariableOpР
conv2d_145/Conv2DConv2D"max_pooling2d_144/MaxPool:output:0(conv2d_145/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЩЩ *
paddingSAME*
strides
2
conv2d_145/Conv2DГ
!conv2d_145/BiasAdd/ReadVariableOpReadVariableOp*conv2d_145_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_145/BiasAdd/ReadVariableOpХ
conv2d_145/BiasAddBiasAddconv2d_145/Conv2D:output:0)conv2d_145/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЩЩ 2
conv2d_145/BiasAddЃ
conv2d_145/ReluReluconv2d_145/BiasAdd:output:0*
T0*1
_output_shapes
:         ЩЩ 2
conv2d_145/Relu═
max_pooling2d_145/MaxPoolMaxPoolconv2d_145/Relu:activations:0*/
_output_shapes
:         }} *
ksize
*
paddingVALID*
strides
2
max_pooling2d_145/MaxPoolХ
 conv2d_146/Conv2D/ReadVariableOpReadVariableOp)conv2d_146_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02"
 conv2d_146/Conv2D/ReadVariableOpЯ
conv2d_146/Conv2DConv2D"max_pooling2d_145/MaxPool:output:0(conv2d_146/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }}@*
paddingSAME*
strides
2
conv2d_146/Conv2DГ
!conv2d_146/BiasAdd/ReadVariableOpReadVariableOp*conv2d_146_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_146/BiasAdd/ReadVariableOp┤
conv2d_146/BiasAddBiasAddconv2d_146/Conv2D:output:0)conv2d_146/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }}@2
conv2d_146/BiasAddЂ
conv2d_146/ReluReluconv2d_146/BiasAdd:output:0*
T0*/
_output_shapes
:         }}@2
conv2d_146/Relu═
max_pooling2d_146/MaxPoolMaxPoolconv2d_146/Relu:activations:0*/
_output_shapes
:         >>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_146/MaxPooly
dropout_37/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *GГќ?2
dropout_37/dropout/ConstИ
dropout_37/dropout/MulMul"max_pooling2d_146/MaxPool:output:0!dropout_37/dropout/Const:output:0*
T0*/
_output_shapes
:         >>@2
dropout_37/dropout/Mulє
dropout_37/dropout/ShapeShape"max_pooling2d_146/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_37/dropout/ShapeП
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*/
_output_shapes
:         >>@*
dtype021
/dropout_37/dropout/random_uniform/RandomUniformІ
!dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *г>2#
!dropout_37/dropout/GreaterEqual/yЫ
dropout_37/dropout/GreaterEqualGreaterEqual8dropout_37/dropout/random_uniform/RandomUniform:output:0*dropout_37/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         >>@2!
dropout_37/dropout/GreaterEqualе
dropout_37/dropout/CastCast#dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         >>@2
dropout_37/dropout/Cast«
dropout_37/dropout/Mul_1Muldropout_37/dropout/Mul:z:0dropout_37/dropout/Cast:y:0*
T0*/
_output_shapes
:         >>@2
dropout_37/dropout/Mul_1u
flatten_48/ConstConst*
_output_shapes
:*
dtype0*
valueB"     ┴ 2
flatten_48/Constа
flatten_48/ReshapeReshapedropout_37/dropout/Mul_1:z:0flatten_48/Const:output:0*
T0*)
_output_shapes
:         ђѓ2
flatten_48/ReshapeФ
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*!
_output_shapes
:ђѓђ*
dtype02 
dense_96/MatMul/ReadVariableOpц
dense_96/MatMulMatMulflatten_48/Reshape:output:0&dense_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_96/MatMulе
dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_96/BiasAdd/ReadVariableOpд
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_96/BiasAddt
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_96/ReluЕ
dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02 
dense_97/MatMul/ReadVariableOpБ
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_97/MatMulД
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_97/BiasAdd/ReadVariableOpЦ
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_97/BiasAddt
IdentityIdentitydense_97/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityп
NoOpNoOp"^conv2d_144/BiasAdd/ReadVariableOp!^conv2d_144/Conv2D/ReadVariableOp"^conv2d_145/BiasAdd/ReadVariableOp!^conv2d_145/Conv2D/ReadVariableOp"^conv2d_146/BiasAdd/ReadVariableOp!^conv2d_146/Conv2D/ReadVariableOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOpF^sequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkipm^sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgt^sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterA^sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip=^sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         ЗЗ: : : : : : : : : : : : : 2F
!conv2d_144/BiasAdd/ReadVariableOp!conv2d_144/BiasAdd/ReadVariableOp2D
 conv2d_144/Conv2D/ReadVariableOp conv2d_144/Conv2D/ReadVariableOp2F
!conv2d_145/BiasAdd/ReadVariableOp!conv2d_145/BiasAdd/ReadVariableOp2D
 conv2d_145/Conv2D/ReadVariableOp conv2d_145/Conv2D/ReadVariableOp2F
!conv2d_146/BiasAdd/ReadVariableOp!conv2d_146/BiasAdd/ReadVariableOp2D
 conv2d_146/Conv2D/ReadVariableOp conv2d_146/Conv2D/ReadVariableOp2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2ј
Esequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkipEsequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkip2▄
lsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlglsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2Ж
ssequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterssequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2ё
@sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip@sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip2|
<sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkip<sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
рR
е

!__inference__wrapped_model_168345
sequential_88_inputQ
7sequential_89_conv2d_144_conv2d_readvariableop_resource:F
8sequential_89_conv2d_144_biasadd_readvariableop_resource:Q
7sequential_89_conv2d_145_conv2d_readvariableop_resource: F
8sequential_89_conv2d_145_biasadd_readvariableop_resource: Q
7sequential_89_conv2d_146_conv2d_readvariableop_resource: @F
8sequential_89_conv2d_146_biasadd_readvariableop_resource:@J
5sequential_89_dense_96_matmul_readvariableop_resource:ђѓђE
6sequential_89_dense_96_biasadd_readvariableop_resource:	ђH
5sequential_89_dense_97_matmul_readvariableop_resource:	ђ
D
6sequential_89_dense_97_biasadd_readvariableop_resource:

identityѕб/sequential_89/conv2d_144/BiasAdd/ReadVariableOpб.sequential_89/conv2d_144/Conv2D/ReadVariableOpб/sequential_89/conv2d_145/BiasAdd/ReadVariableOpб.sequential_89/conv2d_145/Conv2D/ReadVariableOpб/sequential_89/conv2d_146/BiasAdd/ReadVariableOpб.sequential_89/conv2d_146/Conv2D/ReadVariableOpб-sequential_89/dense_96/BiasAdd/ReadVariableOpб,sequential_89/dense_96/MatMul/ReadVariableOpб-sequential_89/dense_97/BiasAdd/ReadVariableOpб,sequential_89/dense_97/MatMul/ReadVariableOpІ
!sequential_89/rescaling_51/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ђђђ;2#
!sequential_89/rescaling_51/Cast/xЈ
#sequential_89/rescaling_51/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_89/rescaling_51/Cast_1/x─
sequential_89/rescaling_51/mulMulsequential_88_input*sequential_89/rescaling_51/Cast/x:output:0*
T0*1
_output_shapes
:         ЗЗ2 
sequential_89/rescaling_51/mulО
sequential_89/rescaling_51/addAddV2"sequential_89/rescaling_51/mul:z:0,sequential_89/rescaling_51/Cast_1/x:output:0*
T0*1
_output_shapes
:         ЗЗ2 
sequential_89/rescaling_51/addЯ
.sequential_89/conv2d_144/Conv2D/ReadVariableOpReadVariableOp7sequential_89_conv2d_144_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.sequential_89/conv2d_144/Conv2D/ReadVariableOpї
sequential_89/conv2d_144/Conv2DConv2D"sequential_89/rescaling_51/add:z:06sequential_89/conv2d_144/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЗЗ*
paddingSAME*
strides
2!
sequential_89/conv2d_144/Conv2DО
/sequential_89/conv2d_144/BiasAdd/ReadVariableOpReadVariableOp8sequential_89_conv2d_144_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_89/conv2d_144/BiasAdd/ReadVariableOpЬ
 sequential_89/conv2d_144/BiasAddBiasAdd(sequential_89/conv2d_144/Conv2D:output:07sequential_89/conv2d_144/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЗЗ2"
 sequential_89/conv2d_144/BiasAddГ
sequential_89/conv2d_144/ReluRelu)sequential_89/conv2d_144/BiasAdd:output:0*
T0*1
_output_shapes
:         ЗЗ2
sequential_89/conv2d_144/Reluщ
'sequential_89/max_pooling2d_144/MaxPoolMaxPool+sequential_89/conv2d_144/Relu:activations:0*1
_output_shapes
:         ЩЩ*
ksize
*
paddingVALID*
strides
2)
'sequential_89/max_pooling2d_144/MaxPoolЯ
.sequential_89/conv2d_145/Conv2D/ReadVariableOpReadVariableOp7sequential_89_conv2d_145_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.sequential_89/conv2d_145/Conv2D/ReadVariableOpџ
sequential_89/conv2d_145/Conv2DConv2D0sequential_89/max_pooling2d_144/MaxPool:output:06sequential_89/conv2d_145/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЩЩ *
paddingSAME*
strides
2!
sequential_89/conv2d_145/Conv2DО
/sequential_89/conv2d_145/BiasAdd/ReadVariableOpReadVariableOp8sequential_89_conv2d_145_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_89/conv2d_145/BiasAdd/ReadVariableOpЬ
 sequential_89/conv2d_145/BiasAddBiasAdd(sequential_89/conv2d_145/Conv2D:output:07sequential_89/conv2d_145/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЩЩ 2"
 sequential_89/conv2d_145/BiasAddГ
sequential_89/conv2d_145/ReluRelu)sequential_89/conv2d_145/BiasAdd:output:0*
T0*1
_output_shapes
:         ЩЩ 2
sequential_89/conv2d_145/Reluэ
'sequential_89/max_pooling2d_145/MaxPoolMaxPool+sequential_89/conv2d_145/Relu:activations:0*/
_output_shapes
:         }} *
ksize
*
paddingVALID*
strides
2)
'sequential_89/max_pooling2d_145/MaxPoolЯ
.sequential_89/conv2d_146/Conv2D/ReadVariableOpReadVariableOp7sequential_89_conv2d_146_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.sequential_89/conv2d_146/Conv2D/ReadVariableOpў
sequential_89/conv2d_146/Conv2DConv2D0sequential_89/max_pooling2d_145/MaxPool:output:06sequential_89/conv2d_146/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }}@*
paddingSAME*
strides
2!
sequential_89/conv2d_146/Conv2DО
/sequential_89/conv2d_146/BiasAdd/ReadVariableOpReadVariableOp8sequential_89_conv2d_146_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential_89/conv2d_146/BiasAdd/ReadVariableOpВ
 sequential_89/conv2d_146/BiasAddBiasAdd(sequential_89/conv2d_146/Conv2D:output:07sequential_89/conv2d_146/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }}@2"
 sequential_89/conv2d_146/BiasAddФ
sequential_89/conv2d_146/ReluRelu)sequential_89/conv2d_146/BiasAdd:output:0*
T0*/
_output_shapes
:         }}@2
sequential_89/conv2d_146/Reluэ
'sequential_89/max_pooling2d_146/MaxPoolMaxPool+sequential_89/conv2d_146/Relu:activations:0*/
_output_shapes
:         >>@*
ksize
*
paddingVALID*
strides
2)
'sequential_89/max_pooling2d_146/MaxPoolЙ
!sequential_89/dropout_37/IdentityIdentity0sequential_89/max_pooling2d_146/MaxPool:output:0*
T0*/
_output_shapes
:         >>@2#
!sequential_89/dropout_37/IdentityЉ
sequential_89/flatten_48/ConstConst*
_output_shapes
:*
dtype0*
valueB"     ┴ 2 
sequential_89/flatten_48/Constп
 sequential_89/flatten_48/ReshapeReshape*sequential_89/dropout_37/Identity:output:0'sequential_89/flatten_48/Const:output:0*
T0*)
_output_shapes
:         ђѓ2"
 sequential_89/flatten_48/ReshapeН
,sequential_89/dense_96/MatMul/ReadVariableOpReadVariableOp5sequential_89_dense_96_matmul_readvariableop_resource*!
_output_shapes
:ђѓђ*
dtype02.
,sequential_89/dense_96/MatMul/ReadVariableOp▄
sequential_89/dense_96/MatMulMatMul)sequential_89/flatten_48/Reshape:output:04sequential_89/dense_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
sequential_89/dense_96/MatMulм
-sequential_89/dense_96/BiasAdd/ReadVariableOpReadVariableOp6sequential_89_dense_96_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02/
-sequential_89/dense_96/BiasAdd/ReadVariableOpя
sequential_89/dense_96/BiasAddBiasAdd'sequential_89/dense_96/MatMul:product:05sequential_89/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2 
sequential_89/dense_96/BiasAddъ
sequential_89/dense_96/ReluRelu'sequential_89/dense_96/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
sequential_89/dense_96/ReluМ
,sequential_89/dense_97/MatMul/ReadVariableOpReadVariableOp5sequential_89_dense_97_matmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02.
,sequential_89/dense_97/MatMul/ReadVariableOp█
sequential_89/dense_97/MatMulMatMul)sequential_89/dense_96/Relu:activations:04sequential_89/dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential_89/dense_97/MatMulЛ
-sequential_89/dense_97/BiasAdd/ReadVariableOpReadVariableOp6sequential_89_dense_97_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_89/dense_97/BiasAdd/ReadVariableOpП
sequential_89/dense_97/BiasAddBiasAdd'sequential_89/dense_97/MatMul:product:05sequential_89/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2 
sequential_89/dense_97/BiasAddѓ
IdentityIdentity'sequential_89/dense_97/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityх
NoOpNoOp0^sequential_89/conv2d_144/BiasAdd/ReadVariableOp/^sequential_89/conv2d_144/Conv2D/ReadVariableOp0^sequential_89/conv2d_145/BiasAdd/ReadVariableOp/^sequential_89/conv2d_145/Conv2D/ReadVariableOp0^sequential_89/conv2d_146/BiasAdd/ReadVariableOp/^sequential_89/conv2d_146/Conv2D/ReadVariableOp.^sequential_89/dense_96/BiasAdd/ReadVariableOp-^sequential_89/dense_96/MatMul/ReadVariableOp.^sequential_89/dense_97/BiasAdd/ReadVariableOp-^sequential_89/dense_97/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЗЗ: : : : : : : : : : 2b
/sequential_89/conv2d_144/BiasAdd/ReadVariableOp/sequential_89/conv2d_144/BiasAdd/ReadVariableOp2`
.sequential_89/conv2d_144/Conv2D/ReadVariableOp.sequential_89/conv2d_144/Conv2D/ReadVariableOp2b
/sequential_89/conv2d_145/BiasAdd/ReadVariableOp/sequential_89/conv2d_145/BiasAdd/ReadVariableOp2`
.sequential_89/conv2d_145/Conv2D/ReadVariableOp.sequential_89/conv2d_145/Conv2D/ReadVariableOp2b
/sequential_89/conv2d_146/BiasAdd/ReadVariableOp/sequential_89/conv2d_146/BiasAdd/ReadVariableOp2`
.sequential_89/conv2d_146/Conv2D/ReadVariableOp.sequential_89/conv2d_146/Conv2D/ReadVariableOp2^
-sequential_89/dense_96/BiasAdd/ReadVariableOp-sequential_89/dense_96/BiasAdd/ReadVariableOp2\
,sequential_89/dense_96/MatMul/ReadVariableOp,sequential_89/dense_96/MatMul/ReadVariableOp2^
-sequential_89/dense_97/BiasAdd/ReadVariableOp-sequential_89/dense_97/BiasAdd/ReadVariableOp2\
,sequential_89/dense_97/MatMul/ReadVariableOp,sequential_89/dense_97/MatMul/ReadVariableOp:f b
1
_output_shapes
:         ЗЗ
-
_user_specified_namesequential_88_input
Н
K
/__inference_random_flip_41_layer_call_fn_170479

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1683532
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ь
Й
.__inference_sequential_88_layer_call_fn_168933
random_flip_41_input
unknown:	
	unknown_0:	
	unknown_1:	
identityѕбStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_41_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1689132
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ЗЗ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
1
_output_shapes
:         ЗЗ
.
_user_specified_namerandom_flip_41_input
В
 
F__inference_conv2d_146_layer_call_and_return_conditional_losses_169054

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }}@*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }}@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         }}@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         }}@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         }} : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         }} 
 
_user_specified_nameinputs
Ч

љ
.__inference_sequential_89_layer_call_fn_169852

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:ђѓђ
	unknown_6:	ђ
	unknown_7:	ђ

	unknown_8:

identityѕбStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_89_layer_call_and_return_conditional_losses_1691102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЗЗ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
е
j
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_168565

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Ь
e
F__inference_dropout_37_layer_call_and_return_conditional_losses_170263

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *GГќ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         >>@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         >>@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *г>2
dropout/GreaterEqual/yк
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         >>@2
dropout/GreaterEqualЄ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         >>@2
dropout/Castѓ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         >>@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         >>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         >>@:W S
/
_output_shapes
:         >>@
 
_user_specified_nameinputs
┤Џ
К
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_168818

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityѕбstateful_uniform/RngReadAndSkipD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1~
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *    2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/ConstЎ
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform/Prodt
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/Cast/xі
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1┘
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkipќ
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stackџ
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1џ
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2╬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_sliceЎ
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcastџ
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stackъ
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1ъ
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2к
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1Ъ
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1а
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/algИ
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         2+
)stateful_uniform/StatelessRandomUniformV2њ
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub»
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         2
stateful_uniform/mulћ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:         2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub/y~
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/subu
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_1/yё
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1Њ
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_2/yѓ
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2Ќ
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_1Ќ
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_3Ќ
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yф
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_5/yѓ
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_6/yё
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6Ў
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_7/yѓ
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7Ў
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_3Ќ
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/addЌ
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y░
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shapeћ
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stackў
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1ў
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2┬
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rotation_matrix/strided_slicey
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cos_2Ъ
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stackБ
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1Б
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2э
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_2Ъ
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stackБ
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1Б
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2э
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2Ї
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2
rotation_matrix/NegЪ
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stackБ
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1Б
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2щ
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_3Ъ
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stackБ
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1Б
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2э
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cos_3Ъ
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stackБ
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1Б
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2э
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5Ъ
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stackБ
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1Б
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2ч
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6ѓ
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1├
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rotation_matrix/zeros/packed
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rotation_matrix/zeros/Constх
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axisе
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shapeѕ
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stackї
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1ї
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2і
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceq
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
transform/fill_value╔
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:         ЗЗ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3ъ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ЗЗ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
љг
Њ
J__inference_random_flip_41_layer_call_and_return_conditional_losses_168448

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityѕб(stateful_uniform_full_int/RngReadAndSkipбCstateless_random_flip_left_right/assert_greater_equal/Assert/AssertбJstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertбOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgбVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterї
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shapeї
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Constй
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prodє
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/xЦ
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1є
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkipе
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stackг
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1г
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2ё
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice┤
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcastг
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack░
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1░
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2Ч
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1║
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1ђ
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg«
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2
stateful_uniform_full_intb

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2

zeros_likeЂ
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:2
stack{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ѕ
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceє
&stateless_random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/Shape┐
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§        26
4stateless_random_flip_left_right/strided_slice/stack║
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6stateless_random_flip_left_right/strided_slice/stack_1║
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2ц
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask20
.stateless_random_flip_left_right/strided_slice▓
6stateless_random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 28
6stateless_random_flip_left_right/assert_positive/ConstГ
Astateless_random_flip_left_right/assert_positive/assert_less/LessLess?stateless_random_flip_left_right/assert_positive/Const:output:07stateless_random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:2C
Astateless_random_flip_left_right/assert_positive/assert_less/Lessм
Bstateless_random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bstateless_random_flip_left_right/assert_positive/assert_less/Constи
@stateless_random_flip_left_right/assert_positive/assert_less/AllAllEstateless_random_flip_left_right/assert_positive/assert_less/Less:z:0Kstateless_random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2B
@stateless_random_flip_left_right/assert_positive/assert_less/AllЂ
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2K
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/ConstЉ
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2S
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0в
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertIstateless_random_flip_left_right/assert_positive/assert_less/All:output:0Zstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2L
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assertљ
%stateless_random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :2'
%stateless_random_flip_left_right/Rank┤
7stateless_random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :29
7stateless_random_flip_left_right/assert_greater_equal/yФ
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual.stateless_random_flip_left_right/Rank:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2D
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqual║
:stateless_random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2<
:stateless_random_flip_left_right/assert_greater_equal/Rank╚
Astateless_random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2C
Astateless_random_flip_left_right/assert_greater_equal/range/start╚
Astateless_random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2C
Astateless_random_flip_left_right/assert_greater_equal/range/deltaЩ
;stateless_random_flip_left_right/assert_greater_equal/rangeRangeJstateless_random_flip_left_right/assert_greater_equal/range/start:output:0Cstateless_random_flip_left_right/assert_greater_equal/Rank:output:0Jstateless_random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2=
;stateless_random_flip_left_right/assert_greater_equal/rangeБ
9stateless_random_flip_left_right/assert_greater_equal/AllAllFstateless_random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Dstateless_random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: 2;
9stateless_random_flip_left_right/assert_greater_equal/AllЗ
Bstateless_random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2D
Bstateless_random_flip_left_right/assert_greater_equal/Assert/ConstЭ
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1ч
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2Ї
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3ё
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0ё
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Є
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Ў
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4њ
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertAssertBstateless_random_flip_left_right/assert_greater_equal/All:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0.stateless_random_flip_left_right/Rank:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0K^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2E
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertЂ
3stateless_random_flip_left_right/control_dependencyIdentityinputsD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*J
_output_shapes8
6:4                                    25
3stateless_random_flip_left_right/control_dependency└
(stateless_random_flip_left_right/Shape_1Shape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2*
(stateless_random_flip_left_right/Shape_1║
6stateless_random_flip_left_right/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6stateless_random_flip_left_right/strided_slice_1/stackЙ
8stateless_random_flip_left_right/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8stateless_random_flip_left_right/strided_slice_1/stack_1Й
8stateless_random_flip_left_right/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8stateless_random_flip_left_right/strided_slice_1/stack_2┤
0stateless_random_flip_left_right/strided_slice_1StridedSlice1stateless_random_flip_left_right/Shape_1:output:0?stateless_random_flip_left_right/strided_slice_1/stack:output:0Astateless_random_flip_left_right/strided_slice_1/stack_1:output:0Astateless_random_flip_left_right/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0stateless_random_flip_left_right/strided_slice_1з
?stateless_random_flip_left_right/stateless_random_uniform/shapePack9stateless_random_flip_left_right/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shape├
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/min├
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2?
=stateless_random_flip_left_right/stateless_random_uniform/maxл
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0D^stateless_random_flip_left_right/assert_greater_equal/Assert/Assert* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterг
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg╩
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:         2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2Х
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subМ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         2?
=stateless_random_flip_left_right/stateless_random_uniform/mulИ
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         2;
9stateless_random_flip_left_right/stateless_random_uniformд
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1д
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2д
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3ѓ
.stateless_random_flip_left_right/Reshape/shapePack9stateless_random_flip_left_right/strided_slice_1:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shapeЉ
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         2*
(stateless_random_flip_left_right/Reshapeк
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         2(
&stateless_random_flip_left_right/Roundг
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axis▓
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*J
_output_shapes8
6:4                                    2,
*stateless_random_flip_left_right/ReverseV2Ѕ
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*J
_output_shapes8
6:4                                    2&
$stateless_random_flip_left_right/mulЋ
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2(
&stateless_random_flip_left_right/sub/xЖ
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         2&
$stateless_random_flip_left_right/subћ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*J
_output_shapes8
6:4                                    2(
&stateless_random_flip_left_right/mul_1ђ
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*J
_output_shapes8
6:4                                    2&
$stateless_random_flip_left_right/addд
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*J
_output_shapes8
6:4                                    2

Identityи
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4                                    : 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2і
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertCstateless_random_flip_left_right/assert_greater_equal/Assert/Assert2ў
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertJstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert2б
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2░
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
­
e
I__inference_sequential_88_layer_call_and_return_conditional_losses_168574

inputs
identity­
random_flip_41/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1685592 
random_flip_41/PartitionedCallЮ
"random_rotation_40/PartitionedCallPartitionedCall'random_flip_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1685652$
"random_rotation_40/PartitionedCallЋ
random_zoom_40/PartitionedCallPartitionedCall+random_rotation_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1685712 
random_zoom_40/PartitionedCallЁ
IdentityIdentity'random_zoom_40/PartitionedCall:output:0*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
╣№
│
I__inference_sequential_88_layer_call_and_return_conditional_losses_170157

inputsN
@random_flip_41_stateful_uniform_full_int_rngreadandskip_resource:	I
;random_rotation_40_stateful_uniform_rngreadandskip_resource:	E
7random_zoom_40_stateful_uniform_rngreadandskip_resource:	
identityѕб7random_flip_41/stateful_uniform_full_int/RngReadAndSkipб^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgбerandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterб2random_rotation_40/stateful_uniform/RngReadAndSkipб.random_zoom_40/stateful_uniform/RngReadAndSkipф
.random_flip_41/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_41/stateful_uniform_full_int/shapeф
.random_flip_41/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 20
.random_flip_41/stateful_uniform_full_int/Constщ
-random_flip_41/stateful_uniform_full_int/ProdProd7random_flip_41/stateful_uniform_full_int/shape:output:07random_flip_41/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2/
-random_flip_41/stateful_uniform_full_int/Prodц
/random_flip_41/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :21
/random_flip_41/stateful_uniform_full_int/Cast/xм
/random_flip_41/stateful_uniform_full_int/Cast_1Cast6random_flip_41/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/random_flip_41/stateful_uniform_full_int/Cast_1Л
7random_flip_41/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip@random_flip_41_stateful_uniform_full_int_rngreadandskip_resource8random_flip_41/stateful_uniform_full_int/Cast/x:output:03random_flip_41/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:29
7random_flip_41/stateful_uniform_full_int/RngReadAndSkipк
<random_flip_41/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<random_flip_41/stateful_uniform_full_int/strided_slice/stack╩
>random_flip_41/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>random_flip_41/stateful_uniform_full_int/strided_slice/stack_1╩
>random_flip_41/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>random_flip_41/stateful_uniform_full_int/strided_slice/stack_2я
6random_flip_41/stateful_uniform_full_int/strided_sliceStridedSlice?random_flip_41/stateful_uniform_full_int/RngReadAndSkip:value:0Erandom_flip_41/stateful_uniform_full_int/strided_slice/stack:output:0Grandom_flip_41/stateful_uniform_full_int/strided_slice/stack_1:output:0Grandom_flip_41/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask28
6random_flip_41/stateful_uniform_full_int/strided_sliceр
0random_flip_41/stateful_uniform_full_int/BitcastBitcast?random_flip_41/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type022
0random_flip_41/stateful_uniform_full_int/Bitcast╩
>random_flip_41/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2@
>random_flip_41/stateful_uniform_full_int/strided_slice_1/stack╬
@random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1╬
@random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2о
8random_flip_41/stateful_uniform_full_int/strided_slice_1StridedSlice?random_flip_41/stateful_uniform_full_int/RngReadAndSkip:value:0Grandom_flip_41/stateful_uniform_full_int/strided_slice_1/stack:output:0Irandom_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Irandom_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2:
8random_flip_41/stateful_uniform_full_int/strided_slice_1у
2random_flip_41/stateful_uniform_full_int/Bitcast_1BitcastArandom_flip_41/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type024
2random_flip_41/stateful_uniform_full_int/Bitcast_1ъ
,random_flip_41/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2.
,random_flip_41/stateful_uniform_full_int/algѕ
(random_flip_41/stateful_uniform_full_intStatelessRandomUniformFullIntV27random_flip_41/stateful_uniform_full_int/shape:output:0;random_flip_41/stateful_uniform_full_int/Bitcast_1:output:09random_flip_41/stateful_uniform_full_int/Bitcast:output:05random_flip_41/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2*
(random_flip_41/stateful_uniform_full_intђ
random_flip_41/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip_41/zeros_likeй
random_flip_41/stackPack1random_flip_41/stateful_uniform_full_int:output:0"random_flip_41/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_flip_41/stackЎ
"random_flip_41/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"random_flip_41/strided_slice/stackЮ
$random_flip_41/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$random_flip_41/strided_slice/stack_1Ю
$random_flip_41/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$random_flip_41/strided_slice/stack_2Р
random_flip_41/strided_sliceStridedSlicerandom_flip_41/stack:output:0+random_flip_41/strided_slice/stack:output:0-random_flip_41/strided_slice/stack_1:output:0-random_flip_41/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip_41/strided_sliceз
Brandom_flip_41/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:         ЗЗ2D
Brandom_flip_41/stateless_random_flip_left_right/control_dependencyж
5random_flip_41/stateless_random_flip_left_right/ShapeShapeKrandom_flip_41/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:27
5random_flip_41/stateless_random_flip_left_right/Shapeн
Crandom_flip_41/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Crandom_flip_41/stateless_random_flip_left_right/strided_slice/stackп
Erandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Erandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_1п
Erandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Erandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_2ѓ
=random_flip_41/stateless_random_flip_left_right/strided_sliceStridedSlice>random_flip_41/stateless_random_flip_left_right/Shape:output:0Lrandom_flip_41/stateless_random_flip_left_right/strided_slice/stack:output:0Nrandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_1:output:0Nrandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=random_flip_41/stateless_random_flip_left_right/strided_sliceъ
Nrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shapePackFrandom_flip_41/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2P
Nrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shapeр
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2N
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/minр
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2N
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/maxи
erandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter%random_flip_41/strided_slice:output:0* 
_output_shapes
::2g
erandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter┘
^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgf^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2`
^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgц
arandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Wrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0krandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0orandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0drandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:         2c
arandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2Ы
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/subSubUrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Urandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2N
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/subЈ
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mulMuljrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Prandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         2N
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mulЗ
Hrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniformAddV2Prandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Urandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         2J
Hrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform─
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2A
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/1─
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2A
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/2─
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2A
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/3┌
=random_flip_41/stateless_random_flip_left_right/Reshape/shapePackFrandom_flip_41/stateless_random_flip_left_right/strided_slice:output:0Hrandom_flip_41/stateless_random_flip_left_right/Reshape/shape/1:output:0Hrandom_flip_41/stateless_random_flip_left_right/Reshape/shape/2:output:0Hrandom_flip_41/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2?
=random_flip_41/stateless_random_flip_left_right/Reshape/shape═
7random_flip_41/stateless_random_flip_left_right/ReshapeReshapeLrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform:z:0Frandom_flip_41/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         29
7random_flip_41/stateless_random_flip_left_right/Reshapeз
5random_flip_41/stateless_random_flip_left_right/RoundRound@random_flip_41/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         27
5random_flip_41/stateless_random_flip_left_right/Round╩
>random_flip_41/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2@
>random_flip_41/stateless_random_flip_left_right/ReverseV2/axisН
9random_flip_41/stateless_random_flip_left_right/ReverseV2	ReverseV2Krandom_flip_41/stateless_random_flip_left_right/control_dependency:output:0Grandom_flip_41/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         ЗЗ2;
9random_flip_41/stateless_random_flip_left_right/ReverseV2г
3random_flip_41/stateless_random_flip_left_right/mulMul9random_flip_41/stateless_random_flip_left_right/Round:y:0Brandom_flip_41/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ЗЗ25
3random_flip_41/stateless_random_flip_left_right/mul│
5random_flip_41/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?27
5random_flip_41/stateless_random_flip_left_right/sub/xд
3random_flip_41/stateless_random_flip_left_right/subSub>random_flip_41/stateless_random_flip_left_right/sub/x:output:09random_flip_41/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         25
3random_flip_41/stateless_random_flip_left_right/subи
5random_flip_41/stateless_random_flip_left_right/mul_1Mul7random_flip_41/stateless_random_flip_left_right/sub:z:0Krandom_flip_41/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         ЗЗ27
5random_flip_41/stateless_random_flip_left_right/mul_1Б
3random_flip_41/stateless_random_flip_left_right/addAddV27random_flip_41/stateless_random_flip_left_right/mul:z:09random_flip_41/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ЗЗ25
3random_flip_41/stateless_random_flip_left_right/addЏ
random_rotation_40/ShapeShape7random_flip_41/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
random_rotation_40/Shapeџ
&random_rotation_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_rotation_40/strided_slice/stackъ
(random_rotation_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_40/strided_slice/stack_1ъ
(random_rotation_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_40/strided_slice/stack_2н
 random_rotation_40/strided_sliceStridedSlice!random_rotation_40/Shape:output:0/random_rotation_40/strided_slice/stack:output:01random_rotation_40/strided_slice/stack_1:output:01random_rotation_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 random_rotation_40/strided_sliceъ
(random_rotation_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_40/strided_slice_1/stackб
*random_rotation_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_40/strided_slice_1/stack_1б
*random_rotation_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_40/strided_slice_1/stack_2я
"random_rotation_40/strided_slice_1StridedSlice!random_rotation_40/Shape:output:01random_rotation_40/strided_slice_1/stack:output:03random_rotation_40/strided_slice_1/stack_1:output:03random_rotation_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_rotation_40/strided_slice_1Ќ
random_rotation_40/CastCast+random_rotation_40/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_40/Castъ
(random_rotation_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_40/strided_slice_2/stackб
*random_rotation_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_40/strided_slice_2/stack_1б
*random_rotation_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_40/strided_slice_2/stack_2я
"random_rotation_40/strided_slice_2StridedSlice!random_rotation_40/Shape:output:01random_rotation_40/strided_slice_2/stack:output:03random_rotation_40/strided_slice_2/stack_1:output:03random_rotation_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_rotation_40/strided_slice_2Џ
random_rotation_40/Cast_1Cast+random_rotation_40/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_40/Cast_1и
)random_rotation_40/stateful_uniform/shapePack)random_rotation_40/strided_slice:output:0*
N*
T0*
_output_shapes
:2+
)random_rotation_40/stateful_uniform/shapeЌ
'random_rotation_40/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_rotation_40/stateful_uniform/minЌ
'random_rotation_40/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_rotation_40/stateful_uniform/maxа
)random_rotation_40/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)random_rotation_40/stateful_uniform/Constт
(random_rotation_40/stateful_uniform/ProdProd2random_rotation_40/stateful_uniform/shape:output:02random_rotation_40/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/stateful_uniform/Prodџ
*random_rotation_40/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*random_rotation_40/stateful_uniform/Cast/x├
*random_rotation_40/stateful_uniform/Cast_1Cast1random_rotation_40/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*random_rotation_40/stateful_uniform/Cast_1И
2random_rotation_40/stateful_uniform/RngReadAndSkipRngReadAndSkip;random_rotation_40_stateful_uniform_rngreadandskip_resource3random_rotation_40/stateful_uniform/Cast/x:output:0.random_rotation_40/stateful_uniform/Cast_1:y:0*
_output_shapes
:24
2random_rotation_40/stateful_uniform/RngReadAndSkip╝
7random_rotation_40/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7random_rotation_40/stateful_uniform/strided_slice/stack└
9random_rotation_40/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9random_rotation_40/stateful_uniform/strided_slice/stack_1└
9random_rotation_40/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9random_rotation_40/stateful_uniform/strided_slice/stack_2└
1random_rotation_40/stateful_uniform/strided_sliceStridedSlice:random_rotation_40/stateful_uniform/RngReadAndSkip:value:0@random_rotation_40/stateful_uniform/strided_slice/stack:output:0Brandom_rotation_40/stateful_uniform/strided_slice/stack_1:output:0Brandom_rotation_40/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask23
1random_rotation_40/stateful_uniform/strided_sliceм
+random_rotation_40/stateful_uniform/BitcastBitcast:random_rotation_40/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02-
+random_rotation_40/stateful_uniform/Bitcast└
9random_rotation_40/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9random_rotation_40/stateful_uniform/strided_slice_1/stack─
;random_rotation_40/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_rotation_40/stateful_uniform/strided_slice_1/stack_1─
;random_rotation_40/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_rotation_40/stateful_uniform/strided_slice_1/stack_2И
3random_rotation_40/stateful_uniform/strided_slice_1StridedSlice:random_rotation_40/stateful_uniform/RngReadAndSkip:value:0Brandom_rotation_40/stateful_uniform/strided_slice_1/stack:output:0Drandom_rotation_40/stateful_uniform/strided_slice_1/stack_1:output:0Drandom_rotation_40/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:25
3random_rotation_40/stateful_uniform/strided_slice_1п
-random_rotation_40/stateful_uniform/Bitcast_1Bitcast<random_rotation_40/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02/
-random_rotation_40/stateful_uniform/Bitcast_1к
@random_rotation_40/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2B
@random_rotation_40/stateful_uniform/StatelessRandomUniformV2/algф
<random_rotation_40/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV22random_rotation_40/stateful_uniform/shape:output:06random_rotation_40/stateful_uniform/Bitcast_1:output:04random_rotation_40/stateful_uniform/Bitcast:output:0Irandom_rotation_40/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         2>
<random_rotation_40/stateful_uniform/StatelessRandomUniformV2я
'random_rotation_40/stateful_uniform/subSub0random_rotation_40/stateful_uniform/max:output:00random_rotation_40/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2)
'random_rotation_40/stateful_uniform/subч
'random_rotation_40/stateful_uniform/mulMulErandom_rotation_40/stateful_uniform/StatelessRandomUniformV2:output:0+random_rotation_40/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         2)
'random_rotation_40/stateful_uniform/mulЯ
#random_rotation_40/stateful_uniformAddV2+random_rotation_40/stateful_uniform/mul:z:00random_rotation_40/stateful_uniform/min:output:0*
T0*#
_output_shapes
:         2%
#random_rotation_40/stateful_uniformЎ
(random_rotation_40/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(random_rotation_40/rotation_matrix/sub/y╩
&random_rotation_40/rotation_matrix/subSubrandom_rotation_40/Cast_1:y:01random_rotation_40/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2(
&random_rotation_40/rotation_matrix/sub«
&random_rotation_40/rotation_matrix/CosCos'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         2(
&random_rotation_40/rotation_matrix/CosЮ
*random_rotation_40/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2,
*random_rotation_40/rotation_matrix/sub_1/yл
(random_rotation_40/rotation_matrix/sub_1Subrandom_rotation_40/Cast_1:y:03random_rotation_40/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/rotation_matrix/sub_1▀
&random_rotation_40/rotation_matrix/mulMul*random_rotation_40/rotation_matrix/Cos:y:0,random_rotation_40/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         2(
&random_rotation_40/rotation_matrix/mul«
&random_rotation_40/rotation_matrix/SinSin'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         2(
&random_rotation_40/rotation_matrix/SinЮ
*random_rotation_40/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2,
*random_rotation_40/rotation_matrix/sub_2/y╬
(random_rotation_40/rotation_matrix/sub_2Subrandom_rotation_40/Cast:y:03random_rotation_40/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/rotation_matrix/sub_2с
(random_rotation_40/rotation_matrix/mul_1Mul*random_rotation_40/rotation_matrix/Sin:y:0,random_rotation_40/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/mul_1с
(random_rotation_40/rotation_matrix/sub_3Sub*random_rotation_40/rotation_matrix/mul:z:0,random_rotation_40/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/sub_3с
(random_rotation_40/rotation_matrix/sub_4Sub*random_rotation_40/rotation_matrix/sub:z:0,random_rotation_40/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/sub_4А
,random_rotation_40/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,random_rotation_40/rotation_matrix/truediv/yШ
*random_rotation_40/rotation_matrix/truedivRealDiv,random_rotation_40/rotation_matrix/sub_4:z:05random_rotation_40/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         2,
*random_rotation_40/rotation_matrix/truedivЮ
*random_rotation_40/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2,
*random_rotation_40/rotation_matrix/sub_5/y╬
(random_rotation_40/rotation_matrix/sub_5Subrandom_rotation_40/Cast:y:03random_rotation_40/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/rotation_matrix/sub_5▓
(random_rotation_40/rotation_matrix/Sin_1Sin'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/Sin_1Ю
*random_rotation_40/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2,
*random_rotation_40/rotation_matrix/sub_6/yл
(random_rotation_40/rotation_matrix/sub_6Subrandom_rotation_40/Cast_1:y:03random_rotation_40/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/rotation_matrix/sub_6т
(random_rotation_40/rotation_matrix/mul_2Mul,random_rotation_40/rotation_matrix/Sin_1:y:0,random_rotation_40/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/mul_2▓
(random_rotation_40/rotation_matrix/Cos_1Cos'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/Cos_1Ю
*random_rotation_40/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2,
*random_rotation_40/rotation_matrix/sub_7/y╬
(random_rotation_40/rotation_matrix/sub_7Subrandom_rotation_40/Cast:y:03random_rotation_40/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/rotation_matrix/sub_7т
(random_rotation_40/rotation_matrix/mul_3Mul,random_rotation_40/rotation_matrix/Cos_1:y:0,random_rotation_40/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/mul_3с
&random_rotation_40/rotation_matrix/addAddV2,random_rotation_40/rotation_matrix/mul_2:z:0,random_rotation_40/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         2(
&random_rotation_40/rotation_matrix/addс
(random_rotation_40/rotation_matrix/sub_8Sub,random_rotation_40/rotation_matrix/sub_5:z:0*random_rotation_40/rotation_matrix/add:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/sub_8Ц
.random_rotation_40/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.random_rotation_40/rotation_matrix/truediv_1/yЧ
,random_rotation_40/rotation_matrix/truediv_1RealDiv,random_rotation_40/rotation_matrix/sub_8:z:07random_rotation_40/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         2.
,random_rotation_40/rotation_matrix/truediv_1Ф
(random_rotation_40/rotation_matrix/ShapeShape'random_rotation_40/stateful_uniform:z:0*
T0*
_output_shapes
:2*
(random_rotation_40/rotation_matrix/Shape║
6random_rotation_40/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_rotation_40/rotation_matrix/strided_slice/stackЙ
8random_rotation_40/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_40/rotation_matrix/strided_slice/stack_1Й
8random_rotation_40/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_40/rotation_matrix/strided_slice/stack_2┤
0random_rotation_40/rotation_matrix/strided_sliceStridedSlice1random_rotation_40/rotation_matrix/Shape:output:0?random_rotation_40/rotation_matrix/strided_slice/stack:output:0Arandom_rotation_40/rotation_matrix/strided_slice/stack_1:output:0Arandom_rotation_40/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_rotation_40/rotation_matrix/strided_slice▓
(random_rotation_40/rotation_matrix/Cos_2Cos'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/Cos_2┼
8random_rotation_40/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_1/stack╔
:random_rotation_40/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_1/stack_1╔
:random_rotation_40/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_1/stack_2ж
2random_rotation_40/rotation_matrix/strided_slice_1StridedSlice,random_rotation_40/rotation_matrix/Cos_2:y:0Arandom_rotation_40/rotation_matrix/strided_slice_1/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_1/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_1▓
(random_rotation_40/rotation_matrix/Sin_2Sin'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/Sin_2┼
8random_rotation_40/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_2/stack╔
:random_rotation_40/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_2/stack_1╔
:random_rotation_40/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_2/stack_2ж
2random_rotation_40/rotation_matrix/strided_slice_2StridedSlice,random_rotation_40/rotation_matrix/Sin_2:y:0Arandom_rotation_40/rotation_matrix/strided_slice_2/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_2/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_2к
&random_rotation_40/rotation_matrix/NegNeg;random_rotation_40/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2(
&random_rotation_40/rotation_matrix/Neg┼
8random_rotation_40/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_3/stack╔
:random_rotation_40/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_3/stack_1╔
:random_rotation_40/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_3/stack_2в
2random_rotation_40/rotation_matrix/strided_slice_3StridedSlice.random_rotation_40/rotation_matrix/truediv:z:0Arandom_rotation_40/rotation_matrix/strided_slice_3/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_3/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_3▓
(random_rotation_40/rotation_matrix/Sin_3Sin'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/Sin_3┼
8random_rotation_40/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_4/stack╔
:random_rotation_40/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_4/stack_1╔
:random_rotation_40/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_4/stack_2ж
2random_rotation_40/rotation_matrix/strided_slice_4StridedSlice,random_rotation_40/rotation_matrix/Sin_3:y:0Arandom_rotation_40/rotation_matrix/strided_slice_4/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_4/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_4▓
(random_rotation_40/rotation_matrix/Cos_3Cos'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/Cos_3┼
8random_rotation_40/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_5/stack╔
:random_rotation_40/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_5/stack_1╔
:random_rotation_40/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_5/stack_2ж
2random_rotation_40/rotation_matrix/strided_slice_5StridedSlice,random_rotation_40/rotation_matrix/Cos_3:y:0Arandom_rotation_40/rotation_matrix/strided_slice_5/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_5/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_5┼
8random_rotation_40/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_6/stack╔
:random_rotation_40/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_6/stack_1╔
:random_rotation_40/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_6/stack_2ь
2random_rotation_40/rotation_matrix/strided_slice_6StridedSlice0random_rotation_40/rotation_matrix/truediv_1:z:0Arandom_rotation_40/rotation_matrix/strided_slice_6/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_6/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_6е
1random_rotation_40/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_rotation_40/rotation_matrix/zeros/packed/1Ј
/random_rotation_40/rotation_matrix/zeros/packedPack9random_rotation_40/rotation_matrix/strided_slice:output:0:random_rotation_40/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:21
/random_rotation_40/rotation_matrix/zeros/packedЦ
.random_rotation_40/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.random_rotation_40/rotation_matrix/zeros/ConstЂ
(random_rotation_40/rotation_matrix/zerosFill8random_rotation_40/rotation_matrix/zeros/packed:output:07random_rotation_40/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2*
(random_rotation_40/rotation_matrix/zerosб
.random_rotation_40/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation_40/rotation_matrix/concat/axisТ
)random_rotation_40/rotation_matrix/concatConcatV2;random_rotation_40/rotation_matrix/strided_slice_1:output:0*random_rotation_40/rotation_matrix/Neg:y:0;random_rotation_40/rotation_matrix/strided_slice_3:output:0;random_rotation_40/rotation_matrix/strided_slice_4:output:0;random_rotation_40/rotation_matrix/strided_slice_5:output:0;random_rotation_40/rotation_matrix/strided_slice_6:output:01random_rotation_40/rotation_matrix/zeros:output:07random_rotation_40/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2+
)random_rotation_40/rotation_matrix/concat»
"random_rotation_40/transform/ShapeShape7random_flip_41/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2$
"random_rotation_40/transform/Shape«
0random_rotation_40/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0random_rotation_40/transform/strided_slice/stack▓
2random_rotation_40/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2random_rotation_40/transform/strided_slice/stack_1▓
2random_rotation_40/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2random_rotation_40/transform/strided_slice/stack_2Ч
*random_rotation_40/transform/strided_sliceStridedSlice+random_rotation_40/transform/Shape:output:09random_rotation_40/transform/strided_slice/stack:output:0;random_rotation_40/transform/strided_slice/stack_1:output:0;random_rotation_40/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*random_rotation_40/transform/strided_sliceЌ
'random_rotation_40/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_rotation_40/transform/fill_value┘
7random_rotation_40/transform/ImageProjectiveTransformV3ImageProjectiveTransformV37random_flip_41/stateless_random_flip_left_right/add:z:02random_rotation_40/rotation_matrix/concat:output:03random_rotation_40/transform/strided_slice:output:00random_rotation_40/transform/fill_value:output:0*1
_output_shapes
:         ЗЗ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR29
7random_rotation_40/transform/ImageProjectiveTransformV3е
random_zoom_40/ShapeShapeLrandom_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_40/Shapeњ
"random_zoom_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"random_zoom_40/strided_slice/stackќ
$random_zoom_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$random_zoom_40/strided_slice/stack_1ќ
$random_zoom_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$random_zoom_40/strided_slice/stack_2╝
random_zoom_40/strided_sliceStridedSlicerandom_zoom_40/Shape:output:0+random_zoom_40/strided_slice/stack:output:0-random_zoom_40/strided_slice/stack_1:output:0-random_zoom_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_40/strided_sliceќ
$random_zoom_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$random_zoom_40/strided_slice_1/stackџ
&random_zoom_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&random_zoom_40/strided_slice_1/stack_1џ
&random_zoom_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&random_zoom_40/strided_slice_1/stack_2к
random_zoom_40/strided_slice_1StridedSlicerandom_zoom_40/Shape:output:0-random_zoom_40/strided_slice_1/stack:output:0/random_zoom_40/strided_slice_1/stack_1:output:0/random_zoom_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
random_zoom_40/strided_slice_1І
random_zoom_40/CastCast'random_zoom_40/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_40/Castќ
$random_zoom_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$random_zoom_40/strided_slice_2/stackџ
&random_zoom_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&random_zoom_40/strided_slice_2/stack_1џ
&random_zoom_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&random_zoom_40/strided_slice_2/stack_2к
random_zoom_40/strided_slice_2StridedSlicerandom_zoom_40/Shape:output:0-random_zoom_40/strided_slice_2/stack:output:0/random_zoom_40/strided_slice_2/stack_1:output:0/random_zoom_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
random_zoom_40/strided_slice_2Ј
random_zoom_40/Cast_1Cast'random_zoom_40/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_40/Cast_1ћ
'random_zoom_40/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'random_zoom_40/stateful_uniform/shape/1П
%random_zoom_40/stateful_uniform/shapePack%random_zoom_40/strided_slice:output:00random_zoom_40/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%random_zoom_40/stateful_uniform/shapeЈ
#random_zoom_40/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2%
#random_zoom_40/stateful_uniform/minЈ
#random_zoom_40/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *═╠ї?2%
#random_zoom_40/stateful_uniform/maxў
%random_zoom_40/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%random_zoom_40/stateful_uniform/ConstН
$random_zoom_40/stateful_uniform/ProdProd.random_zoom_40/stateful_uniform/shape:output:0.random_zoom_40/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2&
$random_zoom_40/stateful_uniform/Prodњ
&random_zoom_40/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom_40/stateful_uniform/Cast/xи
&random_zoom_40/stateful_uniform/Cast_1Cast-random_zoom_40/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2(
&random_zoom_40/stateful_uniform/Cast_1ц
.random_zoom_40/stateful_uniform/RngReadAndSkipRngReadAndSkip7random_zoom_40_stateful_uniform_rngreadandskip_resource/random_zoom_40/stateful_uniform/Cast/x:output:0*random_zoom_40/stateful_uniform/Cast_1:y:0*
_output_shapes
:20
.random_zoom_40/stateful_uniform/RngReadAndSkip┤
3random_zoom_40/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_zoom_40/stateful_uniform/strided_slice/stackИ
5random_zoom_40/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_zoom_40/stateful_uniform/strided_slice/stack_1И
5random_zoom_40/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_zoom_40/stateful_uniform/strided_slice/stack_2е
-random_zoom_40/stateful_uniform/strided_sliceStridedSlice6random_zoom_40/stateful_uniform/RngReadAndSkip:value:0<random_zoom_40/stateful_uniform/strided_slice/stack:output:0>random_zoom_40/stateful_uniform/strided_slice/stack_1:output:0>random_zoom_40/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2/
-random_zoom_40/stateful_uniform/strided_sliceк
'random_zoom_40/stateful_uniform/BitcastBitcast6random_zoom_40/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02)
'random_zoom_40/stateful_uniform/BitcastИ
5random_zoom_40/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5random_zoom_40/stateful_uniform/strided_slice_1/stack╝
7random_zoom_40/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_zoom_40/stateful_uniform/strided_slice_1/stack_1╝
7random_zoom_40/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_zoom_40/stateful_uniform/strided_slice_1/stack_2а
/random_zoom_40/stateful_uniform/strided_slice_1StridedSlice6random_zoom_40/stateful_uniform/RngReadAndSkip:value:0>random_zoom_40/stateful_uniform/strided_slice_1/stack:output:0@random_zoom_40/stateful_uniform/strided_slice_1/stack_1:output:0@random_zoom_40/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:21
/random_zoom_40/stateful_uniform/strided_slice_1╠
)random_zoom_40/stateful_uniform/Bitcast_1Bitcast8random_zoom_40/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02+
)random_zoom_40/stateful_uniform/Bitcast_1Й
<random_zoom_40/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2>
<random_zoom_40/stateful_uniform/StatelessRandomUniformV2/algќ
8random_zoom_40/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2.random_zoom_40/stateful_uniform/shape:output:02random_zoom_40/stateful_uniform/Bitcast_1:output:00random_zoom_40/stateful_uniform/Bitcast:output:0Erandom_zoom_40/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:         2:
8random_zoom_40/stateful_uniform/StatelessRandomUniformV2╬
#random_zoom_40/stateful_uniform/subSub,random_zoom_40/stateful_uniform/max:output:0,random_zoom_40/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2%
#random_zoom_40/stateful_uniform/sub№
#random_zoom_40/stateful_uniform/mulMulArandom_zoom_40/stateful_uniform/StatelessRandomUniformV2:output:0'random_zoom_40/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:         2%
#random_zoom_40/stateful_uniform/mulн
random_zoom_40/stateful_uniformAddV2'random_zoom_40/stateful_uniform/mul:z:0,random_zoom_40/stateful_uniform/min:output:0*
T0*'
_output_shapes
:         2!
random_zoom_40/stateful_uniformz
random_zoom_40/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
random_zoom_40/concat/axisС
random_zoom_40/concatConcatV2#random_zoom_40/stateful_uniform:z:0#random_zoom_40/stateful_uniform:z:0#random_zoom_40/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
random_zoom_40/concatњ
 random_zoom_40/zoom_matrix/ShapeShaperandom_zoom_40/concat:output:0*
T0*
_output_shapes
:2"
 random_zoom_40/zoom_matrix/Shapeф
.random_zoom_40/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.random_zoom_40/zoom_matrix/strided_slice/stack«
0random_zoom_40/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0random_zoom_40/zoom_matrix/strided_slice/stack_1«
0random_zoom_40/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0random_zoom_40/zoom_matrix/strided_slice/stack_2ё
(random_zoom_40/zoom_matrix/strided_sliceStridedSlice)random_zoom_40/zoom_matrix/Shape:output:07random_zoom_40/zoom_matrix/strided_slice/stack:output:09random_zoom_40/zoom_matrix/strided_slice/stack_1:output:09random_zoom_40/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(random_zoom_40/zoom_matrix/strided_sliceЅ
 random_zoom_40/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2"
 random_zoom_40/zoom_matrix/sub/y«
random_zoom_40/zoom_matrix/subSubrandom_zoom_40/Cast_1:y:0)random_zoom_40/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2 
random_zoom_40/zoom_matrix/subЉ
$random_zoom_40/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$random_zoom_40/zoom_matrix/truediv/yК
"random_zoom_40/zoom_matrix/truedivRealDiv"random_zoom_40/zoom_matrix/sub:z:0-random_zoom_40/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2$
"random_zoom_40/zoom_matrix/truediv╣
0random_zoom_40/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            22
0random_zoom_40/zoom_matrix/strided_slice_1/stackй
2random_zoom_40/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           24
2random_zoom_40/zoom_matrix/strided_slice_1/stack_1й
2random_zoom_40/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         24
2random_zoom_40/zoom_matrix/strided_slice_1/stack_2╦
*random_zoom_40/zoom_matrix/strided_slice_1StridedSlicerandom_zoom_40/concat:output:09random_zoom_40/zoom_matrix/strided_slice_1/stack:output:0;random_zoom_40/zoom_matrix/strided_slice_1/stack_1:output:0;random_zoom_40/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2,
*random_zoom_40/zoom_matrix/strided_slice_1Ї
"random_zoom_40/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2$
"random_zoom_40/zoom_matrix/sub_1/x▀
 random_zoom_40/zoom_matrix/sub_1Sub+random_zoom_40/zoom_matrix/sub_1/x:output:03random_zoom_40/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:         2"
 random_zoom_40/zoom_matrix/sub_1К
random_zoom_40/zoom_matrix/mulMul&random_zoom_40/zoom_matrix/truediv:z:0$random_zoom_40/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:         2 
random_zoom_40/zoom_matrix/mulЇ
"random_zoom_40/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2$
"random_zoom_40/zoom_matrix/sub_2/y▓
 random_zoom_40/zoom_matrix/sub_2Subrandom_zoom_40/Cast:y:0+random_zoom_40/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2"
 random_zoom_40/zoom_matrix/sub_2Ћ
&random_zoom_40/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&random_zoom_40/zoom_matrix/truediv_1/y¤
$random_zoom_40/zoom_matrix/truediv_1RealDiv$random_zoom_40/zoom_matrix/sub_2:z:0/random_zoom_40/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2&
$random_zoom_40/zoom_matrix/truediv_1╣
0random_zoom_40/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           22
0random_zoom_40/zoom_matrix/strided_slice_2/stackй
2random_zoom_40/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           24
2random_zoom_40/zoom_matrix/strided_slice_2/stack_1й
2random_zoom_40/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         24
2random_zoom_40/zoom_matrix/strided_slice_2/stack_2╦
*random_zoom_40/zoom_matrix/strided_slice_2StridedSlicerandom_zoom_40/concat:output:09random_zoom_40/zoom_matrix/strided_slice_2/stack:output:0;random_zoom_40/zoom_matrix/strided_slice_2/stack_1:output:0;random_zoom_40/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2,
*random_zoom_40/zoom_matrix/strided_slice_2Ї
"random_zoom_40/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2$
"random_zoom_40/zoom_matrix/sub_3/x▀
 random_zoom_40/zoom_matrix/sub_3Sub+random_zoom_40/zoom_matrix/sub_3/x:output:03random_zoom_40/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2"
 random_zoom_40/zoom_matrix/sub_3═
 random_zoom_40/zoom_matrix/mul_1Mul(random_zoom_40/zoom_matrix/truediv_1:z:0$random_zoom_40/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:         2"
 random_zoom_40/zoom_matrix/mul_1╣
0random_zoom_40/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            22
0random_zoom_40/zoom_matrix/strided_slice_3/stackй
2random_zoom_40/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           24
2random_zoom_40/zoom_matrix/strided_slice_3/stack_1й
2random_zoom_40/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         24
2random_zoom_40/zoom_matrix/strided_slice_3/stack_2╦
*random_zoom_40/zoom_matrix/strided_slice_3StridedSlicerandom_zoom_40/concat:output:09random_zoom_40/zoom_matrix/strided_slice_3/stack:output:0;random_zoom_40/zoom_matrix/strided_slice_3/stack_1:output:0;random_zoom_40/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2,
*random_zoom_40/zoom_matrix/strided_slice_3ў
)random_zoom_40/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)random_zoom_40/zoom_matrix/zeros/packed/1№
'random_zoom_40/zoom_matrix/zeros/packedPack1random_zoom_40/zoom_matrix/strided_slice:output:02random_zoom_40/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'random_zoom_40/zoom_matrix/zeros/packedЋ
&random_zoom_40/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&random_zoom_40/zoom_matrix/zeros/Constр
 random_zoom_40/zoom_matrix/zerosFill0random_zoom_40/zoom_matrix/zeros/packed:output:0/random_zoom_40/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2"
 random_zoom_40/zoom_matrix/zerosю
+random_zoom_40/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+random_zoom_40/zoom_matrix/zeros_1/packed/1ш
)random_zoom_40/zoom_matrix/zeros_1/packedPack1random_zoom_40/zoom_matrix/strided_slice:output:04random_zoom_40/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2+
)random_zoom_40/zoom_matrix/zeros_1/packedЎ
(random_zoom_40/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(random_zoom_40/zoom_matrix/zeros_1/Constж
"random_zoom_40/zoom_matrix/zeros_1Fill2random_zoom_40/zoom_matrix/zeros_1/packed:output:01random_zoom_40/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:         2$
"random_zoom_40/zoom_matrix/zeros_1╣
0random_zoom_40/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           22
0random_zoom_40/zoom_matrix/strided_slice_4/stackй
2random_zoom_40/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           24
2random_zoom_40/zoom_matrix/strided_slice_4/stack_1й
2random_zoom_40/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         24
2random_zoom_40/zoom_matrix/strided_slice_4/stack_2╦
*random_zoom_40/zoom_matrix/strided_slice_4StridedSlicerandom_zoom_40/concat:output:09random_zoom_40/zoom_matrix/strided_slice_4/stack:output:0;random_zoom_40/zoom_matrix/strided_slice_4/stack_1:output:0;random_zoom_40/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2,
*random_zoom_40/zoom_matrix/strided_slice_4ю
+random_zoom_40/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+random_zoom_40/zoom_matrix/zeros_2/packed/1ш
)random_zoom_40/zoom_matrix/zeros_2/packedPack1random_zoom_40/zoom_matrix/strided_slice:output:04random_zoom_40/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2+
)random_zoom_40/zoom_matrix/zeros_2/packedЎ
(random_zoom_40/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(random_zoom_40/zoom_matrix/zeros_2/Constж
"random_zoom_40/zoom_matrix/zeros_2Fill2random_zoom_40/zoom_matrix/zeros_2/packed:output:01random_zoom_40/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:         2$
"random_zoom_40/zoom_matrix/zeros_2њ
&random_zoom_40/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom_40/zoom_matrix/concat/axisэ
!random_zoom_40/zoom_matrix/concatConcatV23random_zoom_40/zoom_matrix/strided_slice_3:output:0)random_zoom_40/zoom_matrix/zeros:output:0"random_zoom_40/zoom_matrix/mul:z:0+random_zoom_40/zoom_matrix/zeros_1:output:03random_zoom_40/zoom_matrix/strided_slice_4:output:0$random_zoom_40/zoom_matrix/mul_1:z:0+random_zoom_40/zoom_matrix/zeros_2:output:0/random_zoom_40/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2#
!random_zoom_40/zoom_matrix/concat╝
random_zoom_40/transform/ShapeShapeLrandom_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2 
random_zoom_40/transform/Shapeд
,random_zoom_40/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,random_zoom_40/transform/strided_slice/stackф
.random_zoom_40/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.random_zoom_40/transform/strided_slice/stack_1ф
.random_zoom_40/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.random_zoom_40/transform/strided_slice/stack_2С
&random_zoom_40/transform/strided_sliceStridedSlice'random_zoom_40/transform/Shape:output:05random_zoom_40/transform/strided_slice/stack:output:07random_zoom_40/transform/strided_slice/stack_1:output:07random_zoom_40/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2(
&random_zoom_40/transform/strided_sliceЈ
#random_zoom_40/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#random_zoom_40/transform/fill_valueо
3random_zoom_40/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Lrandom_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:0*random_zoom_40/zoom_matrix/concat:output:0/random_zoom_40/transform/strided_slice:output:0,random_zoom_40/transform/fill_value:output:0*1
_output_shapes
:         ЗЗ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR25
3random_zoom_40/transform/ImageProjectiveTransformV3Г
IdentityIdentityHrandom_zoom_40/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityи
NoOpNoOp8^random_flip_41/stateful_uniform_full_int/RngReadAndSkip_^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgf^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter3^random_rotation_40/stateful_uniform/RngReadAndSkip/^random_zoom_40/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ЗЗ: : : 2r
7random_flip_41/stateful_uniform_full_int/RngReadAndSkip7random_flip_41/stateful_uniform_full_int/RngReadAndSkip2└
^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2╬
erandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCountererandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2h
2random_rotation_40/stateful_uniform/RngReadAndSkip2random_rotation_40/stateful_uniform/RngReadAndSkip2`
.random_zoom_40/stateful_uniform/RngReadAndSkip.random_zoom_40/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Э
 
F__inference_conv2d_145_layer_call_and_return_conditional_losses_170217

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЩЩ *
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЩЩ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЩЩ 2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЩЩ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЩЩ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЩЩ
 
_user_specified_nameinputs
Э
O
3__inference_random_rotation_40_layer_call_fn_170625

inputs
identityо
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1685652
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Њ
d
F__inference_dropout_37_layer_call_and_return_conditional_losses_169066

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         >>@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         >>@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         >>@:W S
/
_output_shapes
:         >>@
 
_user_specified_nameinputs
Б
Ю
.__inference_sequential_89_layer_call_fn_169133
sequential_88_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:ђѓђ
	unknown_6:	ђ
	unknown_7:	ђ

	unknown_8:

identityѕбStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallsequential_88_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_89_layer_call_and_return_conditional_losses_1691102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЗЗ: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:         ЗЗ
-
_user_specified_namesequential_88_input
љг
Њ
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170412

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityѕб(stateful_uniform_full_int/RngReadAndSkipбCstateless_random_flip_left_right/assert_greater_equal/Assert/AssertбJstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertбOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgбVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterї
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shapeї
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Constй
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prodє
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/xЦ
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1є
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkipе
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stackг
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1г
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2ё
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice┤
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcastг
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack░
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1░
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2Ч
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1║
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1ђ
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg«
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2
stateful_uniform_full_intb

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2

zeros_likeЂ
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:2
stack{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ѕ
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceє
&stateless_random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/Shape┐
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
§        26
4stateless_random_flip_left_right/strided_slice/stack║
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6stateless_random_flip_left_right/strided_slice/stack_1║
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2ц
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask20
.stateless_random_flip_left_right/strided_slice▓
6stateless_random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 28
6stateless_random_flip_left_right/assert_positive/ConstГ
Astateless_random_flip_left_right/assert_positive/assert_less/LessLess?stateless_random_flip_left_right/assert_positive/Const:output:07stateless_random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:2C
Astateless_random_flip_left_right/assert_positive/assert_less/Lessм
Bstateless_random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bstateless_random_flip_left_right/assert_positive/assert_less/Constи
@stateless_random_flip_left_right/assert_positive/assert_less/AllAllEstateless_random_flip_left_right/assert_positive/assert_less/Less:z:0Kstateless_random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2B
@stateless_random_flip_left_right/assert_positive/assert_less/AllЂ
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2K
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/ConstЉ
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2S
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0в
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertIstateless_random_flip_left_right/assert_positive/assert_less/All:output:0Zstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2L
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assertљ
%stateless_random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :2'
%stateless_random_flip_left_right/Rank┤
7stateless_random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :29
7stateless_random_flip_left_right/assert_greater_equal/yФ
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual.stateless_random_flip_left_right/Rank:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2D
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqual║
:stateless_random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2<
:stateless_random_flip_left_right/assert_greater_equal/Rank╚
Astateless_random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2C
Astateless_random_flip_left_right/assert_greater_equal/range/start╚
Astateless_random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2C
Astateless_random_flip_left_right/assert_greater_equal/range/deltaЩ
;stateless_random_flip_left_right/assert_greater_equal/rangeRangeJstateless_random_flip_left_right/assert_greater_equal/range/start:output:0Cstateless_random_flip_left_right/assert_greater_equal/Rank:output:0Jstateless_random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2=
;stateless_random_flip_left_right/assert_greater_equal/rangeБ
9stateless_random_flip_left_right/assert_greater_equal/AllAllFstateless_random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Dstateless_random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: 2;
9stateless_random_flip_left_right/assert_greater_equal/AllЗ
Bstateless_random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2D
Bstateless_random_flip_left_right/assert_greater_equal/Assert/ConstЭ
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1ч
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2Ї
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3ё
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0ё
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Є
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Ў
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4њ
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertAssertBstateless_random_flip_left_right/assert_greater_equal/All:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0.stateless_random_flip_left_right/Rank:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0K^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2E
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertЂ
3stateless_random_flip_left_right/control_dependencyIdentityinputsD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*J
_output_shapes8
6:4                                    25
3stateless_random_flip_left_right/control_dependency└
(stateless_random_flip_left_right/Shape_1Shape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2*
(stateless_random_flip_left_right/Shape_1║
6stateless_random_flip_left_right/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6stateless_random_flip_left_right/strided_slice_1/stackЙ
8stateless_random_flip_left_right/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8stateless_random_flip_left_right/strided_slice_1/stack_1Й
8stateless_random_flip_left_right/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8stateless_random_flip_left_right/strided_slice_1/stack_2┤
0stateless_random_flip_left_right/strided_slice_1StridedSlice1stateless_random_flip_left_right/Shape_1:output:0?stateless_random_flip_left_right/strided_slice_1/stack:output:0Astateless_random_flip_left_right/strided_slice_1/stack_1:output:0Astateless_random_flip_left_right/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0stateless_random_flip_left_right/strided_slice_1з
?stateless_random_flip_left_right/stateless_random_uniform/shapePack9stateless_random_flip_left_right/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shape├
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/min├
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2?
=stateless_random_flip_left_right/stateless_random_uniform/maxл
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0D^stateless_random_flip_left_right/assert_greater_equal/Assert/Assert* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterг
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg╩
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:         2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2Х
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subМ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         2?
=stateless_random_flip_left_right/stateless_random_uniform/mulИ
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         2;
9stateless_random_flip_left_right/stateless_random_uniformд
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1д
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2д
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3ѓ
.stateless_random_flip_left_right/Reshape/shapePack9stateless_random_flip_left_right/strided_slice_1:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shapeЉ
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         2*
(stateless_random_flip_left_right/Reshapeк
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         2(
&stateless_random_flip_left_right/Roundг
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axis▓
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*J
_output_shapes8
6:4                                    2,
*stateless_random_flip_left_right/ReverseV2Ѕ
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*J
_output_shapes8
6:4                                    2&
$stateless_random_flip_left_right/mulЋ
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2(
&stateless_random_flip_left_right/sub/xЖ
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         2&
$stateless_random_flip_left_right/subћ
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*J
_output_shapes8
6:4                                    2(
&stateless_random_flip_left_right/mul_1ђ
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*J
_output_shapes8
6:4                                    2&
$stateless_random_flip_left_right/addд
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*J
_output_shapes8
6:4                                    2

Identityи
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4                                    : 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2і
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertCstateless_random_flip_left_right/assert_greater_equal/Assert/Assert2ў
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertJstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert2б
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2░
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ч8
о
I__inference_sequential_89_layer_call_and_return_conditional_losses_169425
sequential_88_input"
sequential_88_169386:	"
sequential_88_169388:	"
sequential_88_169390:	+
conv2d_144_169394:
conv2d_144_169396:+
conv2d_145_169400: 
conv2d_145_169402: +
conv2d_146_169406: @
conv2d_146_169408:@$
dense_96_169414:ђѓђ
dense_96_169416:	ђ"
dense_97_169419:	ђ

dense_97_169421:

identityѕб"conv2d_144/StatefulPartitionedCallб"conv2d_145/StatefulPartitionedCallб"conv2d_146/StatefulPartitionedCallб dense_96/StatefulPartitionedCallб dense_97/StatefulPartitionedCallб"dropout_37/StatefulPartitionedCallб%sequential_88/StatefulPartitionedCallО
%sequential_88/StatefulPartitionedCallStatefulPartitionedCallsequential_88_inputsequential_88_169386sequential_88_169388sequential_88_169390*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1689132'
%sequential_88/StatefulPartitionedCallњ
rescaling_51/PartitionedCallPartitionedCall.sequential_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_1690052
rescaling_51/PartitionedCallК
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall%rescaling_51/PartitionedCall:output:0conv2d_144_169394conv2d_144_169396*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_144_layer_call_and_return_conditional_losses_1690182$
"conv2d_144/StatefulPartitionedCallъ
!max_pooling2d_144/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЩЩ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_1689592#
!max_pooling2d_144/PartitionedCall╠
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_144/PartitionedCall:output:0conv2d_145_169400conv2d_145_169402*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЩЩ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_145_layer_call_and_return_conditional_losses_1690362$
"conv2d_145/StatefulPartitionedCallю
!max_pooling2d_145/PartitionedCallPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }} * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_1689712#
!max_pooling2d_145/PartitionedCall╩
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_145/PartitionedCall:output:0conv2d_146_169406conv2d_146_169408*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_146_layer_call_and_return_conditional_losses_1690542$
"conv2d_146/StatefulPartitionedCallю
!max_pooling2d_146/PartitionedCallPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_1689832#
!max_pooling2d_146/PartitionedCallъ
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1691792$
"dropout_37/StatefulPartitionedCallЂ
flatten_48/PartitionedCallPartitionedCall+dropout_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђѓ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_48_layer_call_and_return_conditional_losses_1690742
flatten_48/PartitionedCall▓
 dense_96/StatefulPartitionedCallStatefulPartitionedCall#flatten_48/PartitionedCall:output:0dense_96_169414dense_96_169416*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_1690872"
 dense_96/StatefulPartitionedCallи
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_169419dense_97_169421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_1691032"
 dense_97/StatefulPartitionedCallё
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityл
NoOpNoOp#^conv2d_144/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall&^sequential_88/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         ЗЗ: : : : : : : : : : : : : 2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2N
%sequential_88/StatefulPartitionedCall%sequential_88/StatefulPartitionedCall:f b
1
_output_shapes
:         ЗЗ
-
_user_specified_namesequential_88_input
ў
X
.__inference_sequential_88_layer_call_fn_168577
random_flip_41_input
identity▀
PartitionedCallPartitionedCallrandom_flip_41_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1685742
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:g c
1
_output_shapes
:         ЗЗ
.
_user_specified_namerandom_flip_41_input
д
а
+__inference_conv2d_145_layer_call_fn_170226

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЩЩ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_145_layer_call_and_return_conditional_losses_1690362
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЩЩ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЩЩ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЩЩ
 
_user_specified_nameinputs
ш
П
.__inference_sequential_89_layer_call_fn_169883

inputs
unknown:	
	unknown_0:	
	unknown_1:	#
	unknown_2:
	unknown_3:#
	unknown_4: 
	unknown_5: #
	unknown_6: @
	unknown_7:@
	unknown_8:ђѓђ
	unknown_9:	ђ

unknown_10:	ђ


unknown_11:

identityѕбStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_89_layer_call_and_return_conditional_losses_1692872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         ЗЗ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Б
e
I__inference_sequential_88_layer_call_and_return_conditional_losses_169887

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Л4
Ю
I__inference_sequential_89_layer_call_and_return_conditional_losses_169383
sequential_88_input+
conv2d_144_169352:
conv2d_144_169354:+
conv2d_145_169358: 
conv2d_145_169360: +
conv2d_146_169364: @
conv2d_146_169366:@$
dense_96_169372:ђѓђ
dense_96_169374:	ђ"
dense_97_169377:	ђ

dense_97_169379:

identityѕб"conv2d_144/StatefulPartitionedCallб"conv2d_145/StatefulPartitionedCallб"conv2d_146/StatefulPartitionedCallб dense_96/StatefulPartitionedCallб dense_97/StatefulPartitionedCallЩ
sequential_88/PartitionedCallPartitionedCallsequential_88_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1685742
sequential_88/PartitionedCallі
rescaling_51/PartitionedCallPartitionedCall&sequential_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_1690052
rescaling_51/PartitionedCallК
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall%rescaling_51/PartitionedCall:output:0conv2d_144_169352conv2d_144_169354*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_144_layer_call_and_return_conditional_losses_1690182$
"conv2d_144/StatefulPartitionedCallъ
!max_pooling2d_144/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЩЩ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_1689592#
!max_pooling2d_144/PartitionedCall╠
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_144/PartitionedCall:output:0conv2d_145_169358conv2d_145_169360*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЩЩ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_145_layer_call_and_return_conditional_losses_1690362$
"conv2d_145/StatefulPartitionedCallю
!max_pooling2d_145/PartitionedCallPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }} * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_1689712#
!max_pooling2d_145/PartitionedCall╩
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_145/PartitionedCall:output:0conv2d_146_169364conv2d_146_169366*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_conv2d_146_layer_call_and_return_conditional_losses_1690542$
"conv2d_146/StatefulPartitionedCallю
!max_pooling2d_146/PartitionedCallPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_1689832#
!max_pooling2d_146/PartitionedCallє
dropout_37/PartitionedCallPartitionedCall*max_pooling2d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1690662
dropout_37/PartitionedCallщ
flatten_48/PartitionedCallPartitionedCall#dropout_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         ђѓ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *O
fJRH
F__inference_flatten_48_layer_call_and_return_conditional_losses_1690742
flatten_48/PartitionedCall▓
 dense_96/StatefulPartitionedCallStatefulPartitionedCall#flatten_48/PartitionedCall:output:0dense_96_169372dense_96_169374*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_1690872"
 dense_96/StatefulPartitionedCallи
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_169377dense_97_169379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_1691032"
 dense_97/StatefulPartitionedCallё
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

IdentityЃ
NoOpNoOp#^conv2d_144/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЗЗ: : : : : : : : : : 2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall:f b
1
_output_shapes
:         ЗЗ
-
_user_specified_namesequential_88_input
Ш

/__inference_random_flip_41_layer_call_fn_170498

inputs
unknown:	
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1688882
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ЗЗ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Э
 
F__inference_conv2d_144_layer_call_and_return_conditional_losses_170197

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЦ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЗЗ*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpі
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЗЗ2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:         ЗЗ2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ЗЗ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
е
j
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170502

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
В
 
F__inference_conv2d_146_layer_call_and_return_conditional_losses_170237

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }}@*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }}@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         }}@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         }}@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         }} : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         }} 
 
_user_specified_nameinputs
ф
d
H__inference_rescaling_51_layer_call_and_return_conditional_losses_169005

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ђђђ;2
Cast/xY
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2

Cast_1/xf
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:         ЗЗ2
mulk
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:         ЗЗ2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
ЂA
Ф
I__inference_sequential_89_layer_call_and_return_conditional_losses_169506

inputsC
)conv2d_144_conv2d_readvariableop_resource:8
*conv2d_144_biasadd_readvariableop_resource:C
)conv2d_145_conv2d_readvariableop_resource: 8
*conv2d_145_biasadd_readvariableop_resource: C
)conv2d_146_conv2d_readvariableop_resource: @8
*conv2d_146_biasadd_readvariableop_resource:@<
'dense_96_matmul_readvariableop_resource:ђѓђ7
(dense_96_biasadd_readvariableop_resource:	ђ:
'dense_97_matmul_readvariableop_resource:	ђ
6
(dense_97_biasadd_readvariableop_resource:

identityѕб!conv2d_144/BiasAdd/ReadVariableOpб conv2d_144/Conv2D/ReadVariableOpб!conv2d_145/BiasAdd/ReadVariableOpб conv2d_145/Conv2D/ReadVariableOpб!conv2d_146/BiasAdd/ReadVariableOpб conv2d_146/Conv2D/ReadVariableOpбdense_96/BiasAdd/ReadVariableOpбdense_96/MatMul/ReadVariableOpбdense_97/BiasAdd/ReadVariableOpбdense_97/MatMul/ReadVariableOpo
rescaling_51/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *Ђђђ;2
rescaling_51/Cast/xs
rescaling_51/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_51/Cast_1/xЇ
rescaling_51/mulMulinputsrescaling_51/Cast/x:output:0*
T0*1
_output_shapes
:         ЗЗ2
rescaling_51/mulЪ
rescaling_51/addAddV2rescaling_51/mul:z:0rescaling_51/Cast_1/x:output:0*
T0*1
_output_shapes
:         ЗЗ2
rescaling_51/addХ
 conv2d_144/Conv2D/ReadVariableOpReadVariableOp)conv2d_144_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_144/Conv2D/ReadVariableOpн
conv2d_144/Conv2DConv2Drescaling_51/add:z:0(conv2d_144/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЗЗ*
paddingSAME*
strides
2
conv2d_144/Conv2DГ
!conv2d_144/BiasAdd/ReadVariableOpReadVariableOp*conv2d_144_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_144/BiasAdd/ReadVariableOpХ
conv2d_144/BiasAddBiasAddconv2d_144/Conv2D:output:0)conv2d_144/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЗЗ2
conv2d_144/BiasAddЃ
conv2d_144/ReluReluconv2d_144/BiasAdd:output:0*
T0*1
_output_shapes
:         ЗЗ2
conv2d_144/Relu¤
max_pooling2d_144/MaxPoolMaxPoolconv2d_144/Relu:activations:0*1
_output_shapes
:         ЩЩ*
ksize
*
paddingVALID*
strides
2
max_pooling2d_144/MaxPoolХ
 conv2d_145/Conv2D/ReadVariableOpReadVariableOp)conv2d_145_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 conv2d_145/Conv2D/ReadVariableOpР
conv2d_145/Conv2DConv2D"max_pooling2d_144/MaxPool:output:0(conv2d_145/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЩЩ *
paddingSAME*
strides
2
conv2d_145/Conv2DГ
!conv2d_145/BiasAdd/ReadVariableOpReadVariableOp*conv2d_145_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_145/BiasAdd/ReadVariableOpХ
conv2d_145/BiasAddBiasAddconv2d_145/Conv2D:output:0)conv2d_145/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ЩЩ 2
conv2d_145/BiasAddЃ
conv2d_145/ReluReluconv2d_145/BiasAdd:output:0*
T0*1
_output_shapes
:         ЩЩ 2
conv2d_145/Relu═
max_pooling2d_145/MaxPoolMaxPoolconv2d_145/Relu:activations:0*/
_output_shapes
:         }} *
ksize
*
paddingVALID*
strides
2
max_pooling2d_145/MaxPoolХ
 conv2d_146/Conv2D/ReadVariableOpReadVariableOp)conv2d_146_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02"
 conv2d_146/Conv2D/ReadVariableOpЯ
conv2d_146/Conv2DConv2D"max_pooling2d_145/MaxPool:output:0(conv2d_146/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }}@*
paddingSAME*
strides
2
conv2d_146/Conv2DГ
!conv2d_146/BiasAdd/ReadVariableOpReadVariableOp*conv2d_146_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_146/BiasAdd/ReadVariableOp┤
conv2d_146/BiasAddBiasAddconv2d_146/Conv2D:output:0)conv2d_146/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }}@2
conv2d_146/BiasAddЂ
conv2d_146/ReluReluconv2d_146/BiasAdd:output:0*
T0*/
_output_shapes
:         }}@2
conv2d_146/Relu═
max_pooling2d_146/MaxPoolMaxPoolconv2d_146/Relu:activations:0*/
_output_shapes
:         >>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_146/MaxPoolћ
dropout_37/IdentityIdentity"max_pooling2d_146/MaxPool:output:0*
T0*/
_output_shapes
:         >>@2
dropout_37/Identityu
flatten_48/ConstConst*
_output_shapes
:*
dtype0*
valueB"     ┴ 2
flatten_48/Constа
flatten_48/ReshapeReshapedropout_37/Identity:output:0flatten_48/Const:output:0*
T0*)
_output_shapes
:         ђѓ2
flatten_48/ReshapeФ
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*!
_output_shapes
:ђѓђ*
dtype02 
dense_96/MatMul/ReadVariableOpц
dense_96/MatMulMatMulflatten_48/Reshape:output:0&dense_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_96/MatMulе
dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02!
dense_96/BiasAdd/ReadVariableOpд
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_96/BiasAddt
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ2
dense_96/ReluЕ
dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes
:	ђ
*
dtype02 
dense_97/MatMul/ReadVariableOpБ
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_97/MatMulД
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_97/BiasAdd/ReadVariableOpЦ
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense_97/BiasAddt
IdentityIdentitydense_97/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         
2

IdentityЕ
NoOpNoOp"^conv2d_144/BiasAdd/ReadVariableOp!^conv2d_144/Conv2D/ReadVariableOp"^conv2d_145/BiasAdd/ReadVariableOp!^conv2d_145/Conv2D/ReadVariableOp"^conv2d_146/BiasAdd/ReadVariableOp!^conv2d_146/Conv2D/ReadVariableOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:         ЗЗ: : : : : : : : : : 2F
!conv2d_144/BiasAdd/ReadVariableOp!conv2d_144/BiasAdd/ReadVariableOp2D
 conv2d_144/Conv2D/ReadVariableOp conv2d_144/Conv2D/ReadVariableOp2F
!conv2d_145/BiasAdd/ReadVariableOp!conv2d_145/BiasAdd/ReadVariableOp2D
 conv2d_145/Conv2D/ReadVariableOp conv2d_145/Conv2D/ReadVariableOp2F
!conv2d_146/BiasAdd/ReadVariableOp!conv2d_146/BiasAdd/ReadVariableOp2D
 conv2d_146/Conv2D/ReadVariableOp conv2d_146/Conv2D/ReadVariableOp2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Ь
J
.__inference_sequential_88_layer_call_fn_170162

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1685742
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
№
f
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170327

inputs
identity}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ц
f
J__inference_random_flip_41_layer_call_and_return_conditional_losses_168559

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:         ЗЗ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ЗЗ:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
ь
┌
I__inference_sequential_88_layer_call_and_return_conditional_losses_168913

inputs#
random_flip_41_168903:	'
random_rotation_40_168906:	#
random_zoom_40_168909:	
identityѕб&random_flip_41/StatefulPartitionedCallб*random_rotation_40/StatefulPartitionedCallб&random_zoom_40/StatefulPartitionedCallа
&random_flip_41/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_flip_41_168903*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1688882(
&random_flip_41/StatefulPartitionedCall┘
*random_rotation_40/StatefulPartitionedCallStatefulPartitionedCall/random_flip_41/StatefulPartitionedCall:output:0random_rotation_40_168906*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1688182,
*random_rotation_40/StatefulPartitionedCall═
&random_zoom_40/StatefulPartitionedCallStatefulPartitionedCall3random_rotation_40/StatefulPartitionedCall:output:0random_zoom_40_168909*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1686872(
&random_zoom_40/StatefulPartitionedCallћ
IdentityIdentity/random_zoom_40/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identity═
NoOpNoOp'^random_flip_41/StatefulPartitionedCall+^random_rotation_40/StatefulPartitionedCall'^random_zoom_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:         ЗЗ: : : 2P
&random_flip_41/StatefulPartitionedCall&random_flip_41/StatefulPartitionedCall2X
*random_rotation_40/StatefulPartitionedCall*random_rotation_40/StatefulPartitionedCall2P
&random_zoom_40/StatefulPartitionedCall&random_zoom_40/StatefulPartitionedCall:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Сf
ђ
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170474

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identityѕб(stateful_uniform_full_int/RngReadAndSkipбOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgбVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterї
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shapeї
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Constй
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prodє
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/xЦ
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1є
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkipе
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stackг
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1г
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2ё
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice┤
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcastг
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack░
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1░
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2Ч
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1║
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1ђ
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg«
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2
stateful_uniform_full_intb

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2

zeros_likeЂ
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:2
stack{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2ѕ
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceН
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:         ЗЗ25
3stateless_random_flip_left_right/control_dependency╝
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/ShapeХ
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4stateless_random_flip_left_right/strided_slice/stack║
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_1║
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2е
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.stateless_random_flip_left_right/strided_sliceы
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shape├
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/min├
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2?
=stateless_random_flip_left_right/stateless_random_uniform/maxі
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterг
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg╩
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:         2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2Х
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subМ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:         2?
=stateless_random_flip_left_right/stateless_random_uniform/mulИ
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:         2;
9stateless_random_flip_left_right/stateless_random_uniformд
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1д
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2д
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3ђ
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shapeЉ
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:         2*
(stateless_random_flip_left_right/Reshapeк
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:         2(
&stateless_random_flip_left_right/Roundг
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axisЎ
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:         ЗЗ2,
*stateless_random_flip_left_right/ReverseV2­
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:         ЗЗ2&
$stateless_random_flip_left_right/mulЋ
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2(
&stateless_random_flip_left_right/sub/xЖ
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:         2&
$stateless_random_flip_left_right/subч
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:         ЗЗ2(
&stateless_random_flip_left_right/mul_1у
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:         ЗЗ2&
$stateless_random_flip_left_right/addЇ
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityц
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ЗЗ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2б
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2░
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Г
i
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_168959

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ю
Ж
.__inference_sequential_89_layer_call_fn_169347
sequential_88_input
unknown:	
	unknown_0:	
	unknown_1:	#
	unknown_2:
	unknown_3:#
	unknown_4: 
	unknown_5: #
	unknown_6: @
	unknown_7:@
	unknown_8:ђѓђ
	unknown_9:	ђ

unknown_10:	ђ


unknown_11:

identityѕбStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallsequential_88_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_89_layer_call_and_return_conditional_losses_1692872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         ЗЗ: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:         ЗЗ
-
_user_specified_namesequential_88_input
Г
i
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_168983

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┤Џ
К
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170620

inputs6
(stateful_uniform_rngreadandskip_resource:	
identityѕбstateful_uniform/RngReadAndSkipD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2В
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2В
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1~
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *    2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/ConstЎ
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform/Prodt
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/Cast/xі
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1┘
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkipќ
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stackџ
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1џ
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2╬
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_sliceЎ
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcastџ
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stackъ
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1ъ
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2к
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1Ъ
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1а
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/algИ
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:         2+
)stateful_uniform/StatelessRandomUniformV2њ
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub»
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:         2
stateful_uniform/mulћ
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:         2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub/y~
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/subu
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_1/yё
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1Њ
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_2/yѓ
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2Ќ
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_1Ќ
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_3Ќ
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yф
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:         2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_5/yѓ
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_6/yё
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6Ў
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
rotation_matrix/sub_7/yѓ
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7Ў
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/mul_3Ќ
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/addЌ
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y░
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:         2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shapeћ
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stackў
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1ў
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2┬
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rotation_matrix/strided_slicey
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cos_2Ъ
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stackБ
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1Б
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2э
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_2Ъ
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stackБ
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1Б
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2э
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2Ї
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:         2
rotation_matrix/NegЪ
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stackБ
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1Б
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2щ
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Sin_3Ъ
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stackБ
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1Б
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2э
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:         2
rotation_matrix/Cos_3Ъ
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stackБ
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1Б
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2э
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5Ъ
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stackБ
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1Б
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2ч
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6ѓ
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1├
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rotation_matrix/zeros/packed
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rotation_matrix/zeros/Constх
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:         2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axisе
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shapeѕ
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stackї
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1ї
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2і
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceq
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
transform/fill_value╔
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:         ЗЗ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3ъ
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ЗЗ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
Ш

/__inference_random_zoom_40_layer_call_fn_170750

inputs
unknown:	
identityѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1686872
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ЗЗ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
┌

/__inference_random_flip_41_layer_call_fn_170486

inputs
unknown:	
identityѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1684482
StatefulPartitionedCallъ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*J
_output_shapes8
6:4                                    2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4                                    : 22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
 
Ѓ
3__inference_random_rotation_40_layer_call_fn_170632

inputs
unknown:	
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ЗЗ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1688182
StatefulPartitionedCallЁ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ЗЗ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ЗЗ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ЗЗ
 
_user_specified_nameinputs
█
N
2__inference_max_pooling2d_145_layer_call_fn_168977

inputs
identityЬ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *V
fQRO
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_1689712
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*═
serving_default╣
]
sequential_88_inputF
%serving_default_sequential_88_input:0         ЗЗ<
dense_970
StatefulPartitionedCall:0         
tensorflow/serving/predict:┤Ў
Ѕ
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
о_default_save_signature
+О&call_and_return_all_conditional_losses
п__call__"
_tf_keras_sequential
М
layer-0
layer-1
layer-2
trainable_variables
	variables
regularization_losses
	keras_api
+┘&call_and_return_all_conditional_losses
┌__call__"
_tf_keras_sequential
Д
trainable_variables
	variables
regularization_losses
	keras_api
+█&call_and_return_all_conditional_losses
▄__call__"
_tf_keras_layer
й

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
+П&call_and_return_all_conditional_losses
я__call__"
_tf_keras_layer
Д
$trainable_variables
%	variables
&regularization_losses
'	keras_api
+▀&call_and_return_all_conditional_losses
Я__call__"
_tf_keras_layer
й

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
+р&call_and_return_all_conditional_losses
Р__call__"
_tf_keras_layer
Д
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+с&call_and_return_all_conditional_losses
С__call__"
_tf_keras_layer
й

2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
+т&call_and_return_all_conditional_losses
Т__call__"
_tf_keras_layer
Д
8trainable_variables
9	variables
:regularization_losses
;	keras_api
+у&call_and_return_all_conditional_losses
У__call__"
_tf_keras_layer
Д
<trainable_variables
=	variables
>regularization_losses
?	keras_api
+ж&call_and_return_all_conditional_losses
Ж__call__"
_tf_keras_layer
Д
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
+в&call_and_return_all_conditional_losses
В__call__"
_tf_keras_layer
й

Dkernel
Ebias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+ь&call_and_return_all_conditional_losses
Ь__call__"
_tf_keras_layer
й

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+№&call_and_return_all_conditional_losses
­__call__"
_tf_keras_layer
Е
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_ratem┬m├(m─)m┼2mк3mКDm╚Em╔Jm╩Km╦v╠v═(v╬)v¤2vл3vЛDvмEvМJvнKvН"
tf_deprecated_optimizer
f
0
1
(2
)3
24
35
D6
E7
J8
K9"
trackable_list_wrapper
f
0
1
(2
)3
24
35
D6
E7
J8
K9"
trackable_list_wrapper
 "
trackable_list_wrapper
╬
Ulayer_metrics
trainable_variables
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
	variables
regularization_losses
Ymetrics
п__call__
о_default_save_signature
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
-
ыserving_default"
signature_map
▒
Z_rng
[trainable_variables
\	variables
]regularization_losses
^	keras_api
+Ы&call_and_return_all_conditional_losses
з__call__"
_tf_keras_layer
▒
__rng
`trainable_variables
a	variables
bregularization_losses
c	keras_api
+З&call_and_return_all_conditional_losses
ш__call__"
_tf_keras_layer
▒
d_rng
etrainable_variables
f	variables
gregularization_losses
h	keras_api
+Ш&call_and_return_all_conditional_losses
э__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
ilayer_metrics
trainable_variables
jlayer_regularization_losses

klayers
lnon_trainable_variables
	variables
regularization_losses
mmetrics
┌__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
nlayer_metrics
olayer_regularization_losses
trainable_variables

players
qnon_trainable_variables
	variables
regularization_losses
rmetrics
▄__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_144/kernel
:2conv2d_144/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
slayer_metrics
tlayer_regularization_losses
 trainable_variables

ulayers
vnon_trainable_variables
!	variables
"regularization_losses
wmetrics
я__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
xlayer_metrics
ylayer_regularization_losses
$trainable_variables

zlayers
{non_trainable_variables
%	variables
&regularization_losses
|metrics
Я__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
+:) 2conv2d_145/kernel
: 2conv2d_145/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
}layer_metrics
~layer_regularization_losses
*trainable_variables

layers
ђnon_trainable_variables
+	variables
,regularization_losses
Ђmetrics
Р__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ѓlayer_metrics
 Ѓlayer_regularization_losses
.trainable_variables
ёlayers
Ёnon_trainable_variables
/	variables
0regularization_losses
єmetrics
С__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
+:) @2conv2d_146/kernel
:@2conv2d_146/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Єlayer_metrics
 ѕlayer_regularization_losses
4trainable_variables
Ѕlayers
іnon_trainable_variables
5	variables
6regularization_losses
Іmetrics
Т__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
їlayer_metrics
 Їlayer_regularization_losses
8trainable_variables
јlayers
Јnon_trainable_variables
9	variables
:regularization_losses
љmetrics
У__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Љlayer_metrics
 њlayer_regularization_losses
<trainable_variables
Њlayers
ћnon_trainable_variables
=	variables
>regularization_losses
Ћmetrics
Ж__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ќlayer_metrics
 Ќlayer_regularization_losses
@trainable_variables
ўlayers
Ўnon_trainable_variables
A	variables
Bregularization_losses
џmetrics
В__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
$:"ђѓђ2dense_96/kernel
:ђ2dense_96/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
Џlayer_metrics
 юlayer_regularization_losses
Ftrainable_variables
Юlayers
ъnon_trainable_variables
G	variables
Hregularization_losses
Ъmetrics
Ь__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
": 	ђ
2dense_97/kernel
:
2dense_97/bias
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
х
аlayer_metrics
 Аlayer_regularization_losses
Ltrainable_variables
бlayers
Бnon_trainable_variables
M	variables
Nregularization_losses
цmetrics
­__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ц0
д1"
trackable_list_wrapper
/
Д
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
еlayer_metrics
 Еlayer_regularization_losses
[trainable_variables
фlayers
Фnon_trainable_variables
\	variables
]regularization_losses
гmetrics
з__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
/
Г
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
«layer_metrics
 »layer_regularization_losses
`trainable_variables
░layers
▒non_trainable_variables
a	variables
bregularization_losses
▓metrics
ш__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
/
│
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
┤layer_metrics
 хlayer_regularization_losses
etrainable_variables
Хlayers
иnon_trainable_variables
f	variables
gregularization_losses
Иmetrics
э__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

╣total

║count
╗	variables
╝	keras_api"
_tf_keras_metric
c

йtotal

Йcount
┐
_fn_kwargs
└	variables
┴	keras_api"
_tf_keras_metric
:	2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2Variable
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
╣0
║1"
trackable_list_wrapper
.
╗	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
й0
Й1"
trackable_list_wrapper
.
└	variables"
_generic_user_object
0:.2Adam/conv2d_144/kernel/m
": 2Adam/conv2d_144/bias/m
0:. 2Adam/conv2d_145/kernel/m
":  2Adam/conv2d_145/bias/m
0:. @2Adam/conv2d_146/kernel/m
": @2Adam/conv2d_146/bias/m
):'ђѓђ2Adam/dense_96/kernel/m
!:ђ2Adam/dense_96/bias/m
':%	ђ
2Adam/dense_97/kernel/m
 :
2Adam/dense_97/bias/m
0:.2Adam/conv2d_144/kernel/v
": 2Adam/conv2d_144/bias/v
0:. 2Adam/conv2d_145/kernel/v
":  2Adam/conv2d_145/bias/v
0:. @2Adam/conv2d_146/kernel/v
": @2Adam/conv2d_146/bias/v
):'ђѓђ2Adam/dense_96/kernel/v
!:ђ2Adam/dense_96/bias/v
':%	ђ
2Adam/dense_97/kernel/v
 :
2Adam/dense_97/bias/v
ш2Ы
!__inference__wrapped_model_168345╠
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *<б9
7і4
sequential_88_input         ЗЗ
Ы2№
I__inference_sequential_89_layer_call_and_return_conditional_losses_169506
I__inference_sequential_89_layer_call_and_return_conditional_losses_169827
I__inference_sequential_89_layer_call_and_return_conditional_losses_169383
I__inference_sequential_89_layer_call_and_return_conditional_losses_169425└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
.__inference_sequential_89_layer_call_fn_169133
.__inference_sequential_89_layer_call_fn_169852
.__inference_sequential_89_layer_call_fn_169883
.__inference_sequential_89_layer_call_fn_169347└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
I__inference_sequential_88_layer_call_and_return_conditional_losses_169887
I__inference_sequential_88_layer_call_and_return_conditional_losses_170157
I__inference_sequential_88_layer_call_and_return_conditional_losses_168940
I__inference_sequential_88_layer_call_and_return_conditional_losses_168953└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
є2Ѓ
.__inference_sequential_88_layer_call_fn_168577
.__inference_sequential_88_layer_call_fn_170162
.__inference_sequential_88_layer_call_fn_170173
.__inference_sequential_88_layer_call_fn_168933└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
H__inference_rescaling_51_layer_call_and_return_conditional_losses_170181б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_rescaling_51_layer_call_fn_170186б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_conv2d_144_layer_call_and_return_conditional_losses_170197б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_conv2d_144_layer_call_fn_170206б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
х2▓
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_168959Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
џ2Ќ
2__inference_max_pooling2d_144_layer_call_fn_168965Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
­2ь
F__inference_conv2d_145_layer_call_and_return_conditional_losses_170217б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_conv2d_145_layer_call_fn_170226б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
х2▓
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_168971Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
џ2Ќ
2__inference_max_pooling2d_145_layer_call_fn_168977Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
­2ь
F__inference_conv2d_146_layer_call_and_return_conditional_losses_170237б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_conv2d_146_layer_call_fn_170246б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
х2▓
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_168983Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
џ2Ќ
2__inference_max_pooling2d_146_layer_call_fn_168989Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
╩2К
F__inference_dropout_37_layer_call_and_return_conditional_losses_170251
F__inference_dropout_37_layer_call_and_return_conditional_losses_170263┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ћ2Љ
+__inference_dropout_37_layer_call_fn_170268
+__inference_dropout_37_layer_call_fn_170273┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
F__inference_flatten_48_layer_call_and_return_conditional_losses_170279б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Н2м
+__inference_flatten_48_layer_call_fn_170284б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_96_layer_call_and_return_conditional_losses_170295б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_96_layer_call_fn_170304б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_97_layer_call_and_return_conditional_losses_170314б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_97_layer_call_fn_170323б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ОBн
$__inference_signature_wrapper_169458sequential_88_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170327
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170412
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170416
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170474┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
■2ч
/__inference_random_flip_41_layer_call_fn_170479
/__inference_random_flip_41_layer_call_fn_170486
/__inference_random_flip_41_layer_call_fn_170491
/__inference_random_flip_41_layer_call_fn_170498┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
┌2О
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170502
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170620┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ц2А
3__inference_random_rotation_40_layer_call_fn_170625
3__inference_random_rotation_40_layer_call_fn_170632┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
м2¤
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170636
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170738┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ю2Ў
/__inference_random_zoom_40_layer_call_fn_170743
/__inference_random_zoom_40_layer_call_fn_170750┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 »
!__inference__wrapped_model_168345Ѕ
()23DEJKFбC
<б9
7і4
sequential_88_input         ЗЗ
ф "3ф0
.
dense_97"і
dense_97         
║
F__inference_conv2d_144_layer_call_and_return_conditional_losses_170197p9б6
/б,
*і'
inputs         ЗЗ
ф "/б,
%і"
0         ЗЗ
џ њ
+__inference_conv2d_144_layer_call_fn_170206c9б6
/б,
*і'
inputs         ЗЗ
ф ""і         ЗЗ║
F__inference_conv2d_145_layer_call_and_return_conditional_losses_170217p()9б6
/б,
*і'
inputs         ЩЩ
ф "/б,
%і"
0         ЩЩ 
џ њ
+__inference_conv2d_145_layer_call_fn_170226c()9б6
/б,
*і'
inputs         ЩЩ
ф ""і         ЩЩ Х
F__inference_conv2d_146_layer_call_and_return_conditional_losses_170237l237б4
-б*
(і%
inputs         }} 
ф "-б*
#і 
0         }}@
џ ј
+__inference_conv2d_146_layer_call_fn_170246_237б4
-б*
(і%
inputs         }} 
ф " і         }}@Д
D__inference_dense_96_layer_call_and_return_conditional_losses_170295_DE1б.
'б$
"і
inputs         ђѓ
ф "&б#
і
0         ђ
џ 
)__inference_dense_96_layer_call_fn_170304RDE1б.
'б$
"і
inputs         ђѓ
ф "і         ђЦ
D__inference_dense_97_layer_call_and_return_conditional_losses_170314]JK0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         

џ }
)__inference_dense_97_layer_call_fn_170323PJK0б-
&б#
!і
inputs         ђ
ф "і         
Х
F__inference_dropout_37_layer_call_and_return_conditional_losses_170251l;б8
1б.
(і%
inputs         >>@
p 
ф "-б*
#і 
0         >>@
џ Х
F__inference_dropout_37_layer_call_and_return_conditional_losses_170263l;б8
1б.
(і%
inputs         >>@
p
ф "-б*
#і 
0         >>@
џ ј
+__inference_dropout_37_layer_call_fn_170268_;б8
1б.
(і%
inputs         >>@
p 
ф " і         >>@ј
+__inference_dropout_37_layer_call_fn_170273_;б8
1б.
(і%
inputs         >>@
p
ф " і         >>@г
F__inference_flatten_48_layer_call_and_return_conditional_losses_170279b7б4
-б*
(і%
inputs         >>@
ф "'б$
і
0         ђѓ
џ ё
+__inference_flatten_48_layer_call_fn_170284U7б4
-б*
(і%
inputs         >>@
ф "і         ђѓ­
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_168959ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ╚
2__inference_max_pooling2d_144_layer_call_fn_168965ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ­
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_168971ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ╚
2__inference_max_pooling2d_145_layer_call_fn_168977ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ­
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_168983ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ╚
2__inference_max_pooling2d_146_layer_call_fn_168989ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ы
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170327бVбS
LбI
Cі@
inputs4                                    
p 
ф "HбE
>і;
04                                    
џ ш
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170412дДVбS
LбI
Cі@
inputs4                                    
p
ф "HбE
>і;
04                                    
џ Й
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170416p=б:
3б0
*і'
inputs         ЗЗ
p 
ф "/б,
%і"
0         ЗЗ
џ ┬
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170474tД=б:
3б0
*і'
inputs         ЗЗ
p
ф "/б,
%і"
0         ЗЗ
џ ╔
/__inference_random_flip_41_layer_call_fn_170479ЋVбS
LбI
Cі@
inputs4                                    
p 
ф ";і84                                    ═
/__inference_random_flip_41_layer_call_fn_170486ЎДVбS
LбI
Cі@
inputs4                                    
p
ф ";і84                                    ќ
/__inference_random_flip_41_layer_call_fn_170491c=б:
3б0
*і'
inputs         ЗЗ
p 
ф ""і         ЗЗџ
/__inference_random_flip_41_layer_call_fn_170498gД=б:
3б0
*і'
inputs         ЗЗ
p
ф ""і         ЗЗ┬
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170502p=б:
3б0
*і'
inputs         ЗЗ
p 
ф "/б,
%і"
0         ЗЗ
џ к
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170620tГ=б:
3б0
*і'
inputs         ЗЗ
p
ф "/б,
%і"
0         ЗЗ
џ џ
3__inference_random_rotation_40_layer_call_fn_170625c=б:
3б0
*і'
inputs         ЗЗ
p 
ф ""і         ЗЗъ
3__inference_random_rotation_40_layer_call_fn_170632gГ=б:
3б0
*і'
inputs         ЗЗ
p
ф ""і         ЗЗЙ
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170636p=б:
3б0
*і'
inputs         ЗЗ
p 
ф "/б,
%і"
0         ЗЗ
џ ┬
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170738t│=б:
3б0
*і'
inputs         ЗЗ
p
ф "/б,
%і"
0         ЗЗ
џ ќ
/__inference_random_zoom_40_layer_call_fn_170743c=б:
3б0
*і'
inputs         ЗЗ
p 
ф ""і         ЗЗџ
/__inference_random_zoom_40_layer_call_fn_170750g│=б:
3б0
*і'
inputs         ЗЗ
p
ф ""і         ЗЗИ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_170181l9б6
/б,
*і'
inputs         ЗЗ
ф "/б,
%і"
0         ЗЗ
џ љ
-__inference_rescaling_51_layer_call_fn_170186_9б6
/б,
*і'
inputs         ЗЗ
ф ""і         ЗЗл
I__inference_sequential_88_layer_call_and_return_conditional_losses_168940ѓOбL
EбB
8і5
random_flip_41_input         ЗЗ
p 

 
ф "/б,
%і"
0         ЗЗ
џ п
I__inference_sequential_88_layer_call_and_return_conditional_losses_168953іДГ│OбL
EбB
8і5
random_flip_41_input         ЗЗ
p

 
ф "/б,
%і"
0         ЗЗ
џ ┴
I__inference_sequential_88_layer_call_and_return_conditional_losses_169887tAб>
7б4
*і'
inputs         ЗЗ
p 

 
ф "/б,
%і"
0         ЗЗ
џ ╔
I__inference_sequential_88_layer_call_and_return_conditional_losses_170157|ДГ│Aб>
7б4
*і'
inputs         ЗЗ
p

 
ф "/б,
%і"
0         ЗЗ
џ Д
.__inference_sequential_88_layer_call_fn_168577uOбL
EбB
8і5
random_flip_41_input         ЗЗ
p 

 
ф ""і         ЗЗ»
.__inference_sequential_88_layer_call_fn_168933}ДГ│OбL
EбB
8і5
random_flip_41_input         ЗЗ
p

 
ф ""і         ЗЗЎ
.__inference_sequential_88_layer_call_fn_170162gAб>
7б4
*і'
inputs         ЗЗ
p 

 
ф ""і         ЗЗА
.__inference_sequential_88_layer_call_fn_170173oДГ│Aб>
7б4
*і'
inputs         ЗЗ
p

 
ф ""і         ЗЗЛ
I__inference_sequential_89_layer_call_and_return_conditional_losses_169383Ѓ
()23DEJKNбK
DбA
7і4
sequential_88_input         ЗЗ
p 

 
ф "%б"
і
0         

џ О
I__inference_sequential_89_layer_call_and_return_conditional_losses_169425ЅДГ│()23DEJKNбK
DбA
7і4
sequential_88_input         ЗЗ
p

 
ф "%б"
і
0         

џ ├
I__inference_sequential_89_layer_call_and_return_conditional_losses_169506v
()23DEJKAб>
7б4
*і'
inputs         ЗЗ
p 

 
ф "%б"
і
0         

џ ╔
I__inference_sequential_89_layer_call_and_return_conditional_losses_169827|ДГ│()23DEJKAб>
7б4
*і'
inputs         ЗЗ
p

 
ф "%б"
і
0         

џ е
.__inference_sequential_89_layer_call_fn_169133v
()23DEJKNбK
DбA
7і4
sequential_88_input         ЗЗ
p 

 
ф "і         
«
.__inference_sequential_89_layer_call_fn_169347|ДГ│()23DEJKNбK
DбA
7і4
sequential_88_input         ЗЗ
p

 
ф "і         
Џ
.__inference_sequential_89_layer_call_fn_169852i
()23DEJKAб>
7б4
*і'
inputs         ЗЗ
p 

 
ф "і         
А
.__inference_sequential_89_layer_call_fn_169883oДГ│()23DEJKAб>
7б4
*і'
inputs         ЗЗ
p

 
ф "і         
╔
$__inference_signature_wrapper_169458а
()23DEJK]бZ
б 
SфP
N
sequential_88_input7і4
sequential_88_input         ЗЗ"3ф0
.
dense_97"і
dense_97         
