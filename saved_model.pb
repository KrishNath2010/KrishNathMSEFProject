ñÔ
¦
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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

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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02unknown8¥

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

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

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
shape:* 
shared_namedense_96/kernel
v
#dense_96/kernel/Read/ReadVariableOpReadVariableOpdense_96/kernel*!
_output_shapes
:*
dtype0
s
dense_96/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_96/bias
l
!dense_96/bias/Read/ReadVariableOpReadVariableOpdense_96/bias*
_output_shapes	
:*
dtype0
{
dense_97/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
* 
shared_namedense_97/kernel
t
#dense_97/kernel/Read/ReadVariableOpReadVariableOpdense_97/kernel*
_output_shapes
:	
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

Adam/conv2d_144/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_144/kernel/m

,Adam/conv2d_144/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_144/kernel/m*&
_output_shapes
:*
dtype0

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

Adam/conv2d_145/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_145/kernel/m

,Adam/conv2d_145/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_145/kernel/m*&
_output_shapes
: *
dtype0

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

Adam/conv2d_146/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_146/kernel/m

,Adam/conv2d_146/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_146/kernel/m*&
_output_shapes
: @*
dtype0

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

Adam/dense_96/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_96/kernel/m

*Adam/dense_96/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_96/kernel/m*!
_output_shapes
:*
dtype0

Adam/dense_96/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_96/bias/m
z
(Adam/dense_96/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_96/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_97/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*'
shared_nameAdam/dense_97/kernel/m

*Adam/dense_97/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_97/kernel/m*
_output_shapes
:	
*
dtype0

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

Adam/conv2d_144/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_144/kernel/v

,Adam/conv2d_144/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_144/kernel/v*&
_output_shapes
:*
dtype0

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

Adam/conv2d_145/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_145/kernel/v

,Adam/conv2d_145/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_145/kernel/v*&
_output_shapes
: *
dtype0

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

Adam/conv2d_146/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameAdam/conv2d_146/kernel/v

,Adam/conv2d_146/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_146/kernel/v*&
_output_shapes
: @*
dtype0

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

Adam/dense_96/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_96/kernel/v

*Adam/dense_96/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_96/kernel/v*!
_output_shapes
:*
dtype0

Adam/dense_96/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_96/bias/v
z
(Adam/dense_96/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_96/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_97/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*'
shared_nameAdam/dense_97/kernel/v

*Adam/dense_97/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_97/kernel/v*
_output_shapes
:	
*
dtype0

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
àQ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Q
valueQBQ BQ

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

Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_ratemÂmÃ(mÄ)mÅ2mÆ3mÇDmÈEmÉJmÊKmËvÌvÍ(vÎ)vÏ2vÐ3vÑDvÒEvÓJvÔKvÕ
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
­
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
­
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
­
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
­
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
­
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
¯
}layer_metrics
~layer_regularization_losses
*trainable_variables

layers
non_trainable_variables
+	variables
,regularization_losses
metrics
 
 
 
²
layer_metrics
 layer_regularization_losses
.trainable_variables
layers
non_trainable_variables
/	variables
0regularization_losses
metrics
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
²
layer_metrics
 layer_regularization_losses
4trainable_variables
layers
non_trainable_variables
5	variables
6regularization_losses
metrics
 
 
 
²
layer_metrics
 layer_regularization_losses
8trainable_variables
layers
non_trainable_variables
9	variables
:regularization_losses
metrics
 
 
 
²
layer_metrics
 layer_regularization_losses
<trainable_variables
layers
non_trainable_variables
=	variables
>regularization_losses
metrics
 
 
 
²
layer_metrics
 layer_regularization_losses
@trainable_variables
layers
non_trainable_variables
A	variables
Bregularization_losses
metrics
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
²
layer_metrics
 layer_regularization_losses
Ftrainable_variables
layers
non_trainable_variables
G	variables
Hregularization_losses
metrics
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
²
 layer_metrics
 ¡layer_regularization_losses
Ltrainable_variables
¢layers
£non_trainable_variables
M	variables
Nregularization_losses
¤metrics
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
¥0
¦1

§
_state_var
 
 
 
²
¨layer_metrics
 ©layer_regularization_losses
[trainable_variables
ªlayers
«non_trainable_variables
\	variables
]regularization_losses
¬metrics

­
_state_var
 
 
 
²
®layer_metrics
 ¯layer_regularization_losses
`trainable_variables
°layers
±non_trainable_variables
a	variables
bregularization_losses
²metrics

³
_state_var
 
 
 
²
´layer_metrics
 µlayer_regularization_losses
etrainable_variables
¶layers
·non_trainable_variables
f	variables
gregularization_losses
¸metrics
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

¹total

ºcount
»	variables
¼	keras_api
I

½total

¾count
¿
_fn_kwargs
À	variables
Á	keras_api
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
¹0
º1

»	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

½0
¾1

À	variables
~
VARIABLE_VALUEAdam/conv2d_144/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_144/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_145/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_145/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
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
~
VARIABLE_VALUEAdam/conv2d_144/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_144/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_145/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_145/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
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

#serving_default_sequential_88_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿôô
þ
StatefulPartitionedCallStatefulPartitionedCall#serving_default_sequential_88_inputconv2d_144/kernelconv2d_144/biasconv2d_145/kernelconv2d_145/biasconv2d_146/kernelconv2d_146/biasdense_96/kerneldense_96/biasdense_97/kerneldense_97/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_169458
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¤
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
GPU 2J 8 *(
f#R!
__inference__traced_save_170914
×
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_171050´ä
¤
f
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_168571

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ª4

I__inference_sequential_89_layer_call_and_return_conditional_losses_169110

inputs+
conv2d_144_169019:
conv2d_144_169021:+
conv2d_145_169037: 
conv2d_145_169039: +
conv2d_146_169055: @
conv2d_146_169057:@$
dense_96_169088:
dense_96_169090:	"
dense_97_169104:	

dense_97_169106:

identity¢"conv2d_144/StatefulPartitionedCall¢"conv2d_145/StatefulPartitionedCall¢"conv2d_146/StatefulPartitionedCall¢ dense_96/StatefulPartitionedCall¢ dense_97/StatefulPartitionedCallí
sequential_88/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1685742
sequential_88/PartitionedCall
rescaling_51/PartitionedCallPartitionedCall&sequential_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_1690052
rescaling_51/PartitionedCallÇ
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall%rescaling_51/PartitionedCall:output:0conv2d_144_169019conv2d_144_169021*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_144_layer_call_and_return_conditional_losses_1690182$
"conv2d_144/StatefulPartitionedCall
!max_pooling2d_144/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_1689592#
!max_pooling2d_144/PartitionedCallÌ
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_144/PartitionedCall:output:0conv2d_145_169037conv2d_145_169039*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_145_layer_call_and_return_conditional_losses_1690362$
"conv2d_145/StatefulPartitionedCall
!max_pooling2d_145/PartitionedCallPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}} * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_1689712#
!max_pooling2d_145/PartitionedCallÊ
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_145/PartitionedCall:output:0conv2d_146_169055conv2d_146_169057*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_146_layer_call_and_return_conditional_losses_1690542$
"conv2d_146/StatefulPartitionedCall
!max_pooling2d_146/PartitionedCallPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_1689832#
!max_pooling2d_146/PartitionedCall
dropout_37/PartitionedCallPartitionedCall*max_pooling2d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1690662
dropout_37/PartitionedCallù
flatten_48/PartitionedCallPartitionedCall#dropout_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_48_layer_call_and_return_conditional_losses_1690742
flatten_48/PartitionedCall²
 dense_96/StatefulPartitionedCallStatefulPartitionedCall#flatten_48/PartitionedCall:output:0dense_96_169088dense_96_169090*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_1690872"
 dense_96/StatefulPartitionedCall·
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_169104dense_97_169106*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_1691032"
 dense_97/StatefulPartitionedCall
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity
NoOpNoOp#^conv2d_144/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : 2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
û

)__inference_dense_96_layer_call_fn_170304

inputs
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_1690872
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
b
F__inference_flatten_48_layer_call_and_return_conditional_losses_170279

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ Á 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs

s
I__inference_sequential_88_layer_call_and_return_conditional_losses_168940
random_flip_41_input
identityþ
random_flip_41/PartitionedCallPartitionedCallrandom_flip_41_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1685592 
random_flip_41/PartitionedCall
"random_rotation_40/PartitionedCallPartitionedCall'random_flip_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1685652$
"random_rotation_40/PartitionedCall
random_zoom_40/PartitionedCallPartitionedCall+random_rotation_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1685712 
random_zoom_40/PartitionedCall
IdentityIdentity'random_zoom_40/PartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:g c
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
.
_user_specified_namerandom_flip_41_input
ª

ö
D__inference_dense_97_layer_call_and_return_conditional_losses_170314

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷´
ç
"__inference__traced_restore_171050
file_prefix<
"assignvariableop_conv2d_144_kernel:0
"assignvariableop_1_conv2d_144_bias:>
$assignvariableop_2_conv2d_145_kernel: 0
"assignvariableop_3_conv2d_145_bias: >
$assignvariableop_4_conv2d_146_kernel: @0
"assignvariableop_5_conv2d_146_bias:@7
"assignvariableop_6_dense_96_kernel:/
 assignvariableop_7_dense_96_bias:	5
"assignvariableop_8_dense_97_kernel:	
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
*assignvariableop_28_adam_dense_96_kernel_m:7
(assignvariableop_29_adam_dense_96_bias_m:	=
*assignvariableop_30_adam_dense_97_kernel_m:	
6
(assignvariableop_31_adam_dense_97_bias_m:
F
,assignvariableop_32_adam_conv2d_144_kernel_v:8
*assignvariableop_33_adam_conv2d_144_bias_v:F
,assignvariableop_34_adam_conv2d_145_kernel_v: 8
*assignvariableop_35_adam_conv2d_145_bias_v: F
,assignvariableop_36_adam_conv2d_146_kernel_v: @8
*assignvariableop_37_adam_conv2d_146_bias_v:@?
*assignvariableop_38_adam_dense_96_kernel_v:7
(assignvariableop_39_adam_dense_96_bias_v:	=
*assignvariableop_40_adam_dense_97_kernel_v:	
6
(assignvariableop_41_adam_dense_97_bias_v:

identity_43¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9º
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*Æ
value¼B¹+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesä
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Â
_output_shapes¯
¬:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+				2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¡
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_144_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_144_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_145_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_145_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4©
AssignVariableOp_4AssignVariableOp$assignvariableop_4_conv2d_146_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_146_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_96_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_96_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_97_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_97_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10¥
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11§
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¦
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14®
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_15¤
AssignVariableOp_15AssignVariableOpassignvariableop_15_variableIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16¦
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_17¦
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¡
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¡
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21£
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22´
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_conv2d_144_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23²
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_144_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24´
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_conv2d_145_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_145_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26´
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_conv2d_146_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27²
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_146_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28²
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_dense_96_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29°
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_dense_96_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30²
AssignVariableOp_30AssignVariableOp*assignvariableop_30_adam_dense_97_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31°
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_dense_97_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32´
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_conv2d_144_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33²
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_144_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34´
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_conv2d_145_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35²
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_145_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36´
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_conv2d_146_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37²
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_146_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38²
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_dense_96_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39°
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_dense_96_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40²
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_dense_97_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41°
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_dense_97_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_419
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpú
Identity_42Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_42f
Identity_43IdentityIdentity_42:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_43â
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
ð
K
/__inference_random_flip_41_layer_call_fn_170491

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1685592
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ª

ö
D__inference_dense_97_layer_call_and_return_conditional_losses_169103

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
f
J__inference_random_flip_41_layer_call_and_return_conditional_losses_168353

inputs
identity}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
e
F__inference_dropout_37_layer_call_and_return_conditional_losses_169179

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *G­?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬>2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs
ô
Ã
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170738

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkipD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stateful_uniform/shape/1¡
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
 *ÍÌ?2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/Const
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
stateful_uniform/Cast/x
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1Ù
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkip
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stack
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2Î
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_slice
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stack
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2Æ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1 
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg¼
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateful_uniform/StatelessRandomUniformV2
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub³
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concate
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
zoom_matrix/Shape
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
zoom_matrix/strided_slice/stack
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_1
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_2ª
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
 *  ?2
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
zoom_matrix/truediv/y
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_1/stack
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_1/stack_1
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_1/stack_2ñ
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
 *  ?2
zoom_matrix/sub_1/x£
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/sub_1
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/mulo
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
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
zoom_matrix/truediv_1/y
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv_1
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_2/stack
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_2/stack_1
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_2/stack_2ñ
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
 *  ?2
zoom_matrix/sub_3/x£
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/sub_3
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/mul_1
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_3/stack
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_3/stack_1
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_3/stack_2ñ
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
zoom_matrix/zeros/packed/1³
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
zoom_matrix/zeros/Const¥
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros~
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/packed/1¹
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
zoom_matrix/zeros_1/Const­
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros_1
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_4/stack
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_4/stack_1
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_4/stack_2ñ
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
zoom_matrix/zeros_2/packed/1¹
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
zoom_matrix/zeros_2/Const­
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros_2t
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/concat/axisá
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
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
transform/fill_valueÅ
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿôô: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
¤
f
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170636

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ñ


$__inference_signature_wrapper_169458
sequential_88_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:
	unknown_6:	
	unknown_7:	

	unknown_8:

identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallsequential_88_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_1683452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
1:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
-
_user_specified_namesequential_88_input
­
i
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_168971

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
F__inference_dropout_37_layer_call_and_return_conditional_losses_170251

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs
Ã
°
.__inference_sequential_88_layer_call_fn_170173

inputs
unknown:	
	unknown_0:	
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1689132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿôô: : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs

ù
D__inference_dense_96_layer_call_and_return_conditional_losses_169087

inputs3
matmul_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
ÿ
F__inference_conv2d_145_layer_call_and_return_conditional_losses_169036

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿúú: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
äf

J__inference_random_flip_41_layer_call_and_return_conditional_losses_168888

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkip¢Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shape
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Const½
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prod
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/x¥
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkip¨
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stack¬
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1¬
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice´
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcast¬
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack°
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1°
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2ü
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1º
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg®
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

zeros_like
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
strided_slice/stack_2
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceÕ
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô25
3stateless_random_flip_left_right/control_dependency¼
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/Shape¶
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4stateless_random_flip_left_right/strided_slice/stackº
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_1º
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2¨
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.stateless_random_flip_left_right/strided_sliceñ
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shapeÃ
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/minÃ
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2?
=stateless_random_flip_left_right/stateless_random_uniform/max
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¬
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgÊ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2¶
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subÓ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=stateless_random_flip_left_right/stateless_random_uniform/mul¸
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9stateless_random_flip_left_right/stateless_random_uniform¦
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1¦
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2¦
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shape
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(stateless_random_flip_left_right/ReshapeÆ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_left_right/Round¬
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axis
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2,
*stateless_random_flip_left_right/ReverseV2ð
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2&
$stateless_random_flip_left_right/mul
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&stateless_random_flip_left_right/sub/xê
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stateless_random_flip_left_right/subû
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2(
&stateless_random_flip_left_right/mul_1ç
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2&
$stateless_random_flip_left_right/add
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity¤
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿôô: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2¢
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ô
Ã
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_168687

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkipD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stateful_uniform/shape/1¡
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
 *ÍÌ?2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/Const
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
stateful_uniform/Cast/x
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1Ù
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkip
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stack
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2Î
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_slice
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stack
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2Æ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1 
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg¼
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateful_uniform/StatelessRandomUniformV2
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub³
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2stateful_uniform:z:0stateful_uniform:z:0concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concate
zoom_matrix/ShapeShapeconcat:output:0*
T0*
_output_shapes
:2
zoom_matrix/Shape
zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2!
zoom_matrix/strided_slice/stack
!zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_1
!zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!zoom_matrix/strided_slice/stack_2ª
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
 *  ?2
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
zoom_matrix/truediv/y
zoom_matrix/truedivRealDivzoom_matrix/sub:z:0zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv
!zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_1/stack
#zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_1/stack_1
#zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_1/stack_2ñ
zoom_matrix/strided_slice_1StridedSliceconcat:output:0*zoom_matrix/strided_slice_1/stack:output:0,zoom_matrix/strided_slice_1/stack_1:output:0,zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
 *  ?2
zoom_matrix/sub_1/x£
zoom_matrix/sub_1Subzoom_matrix/sub_1/x:output:0$zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/sub_1
zoom_matrix/mulMulzoom_matrix/truediv:z:0zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/mulo
zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
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
zoom_matrix/truediv_1/y
zoom_matrix/truediv_1RealDivzoom_matrix/sub_2:z:0 zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2
zoom_matrix/truediv_1
!zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_2/stack
#zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_2/stack_1
#zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_2/stack_2ñ
zoom_matrix/strided_slice_2StridedSliceconcat:output:0*zoom_matrix/strided_slice_2/stack:output:0,zoom_matrix/strided_slice_2/stack_1:output:0,zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
 *  ?2
zoom_matrix/sub_3/x£
zoom_matrix/sub_3Subzoom_matrix/sub_3/x:output:0$zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/sub_3
zoom_matrix/mul_1Mulzoom_matrix/truediv_1:z:0zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/mul_1
!zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2#
!zoom_matrix/strided_slice_3/stack
#zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_3/stack_1
#zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_3/stack_2ñ
zoom_matrix/strided_slice_3StridedSliceconcat:output:0*zoom_matrix/strided_slice_3/stack:output:0,zoom_matrix/strided_slice_3/stack_1:output:0,zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
zoom_matrix/zeros/packed/1³
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
zoom_matrix/zeros/Const¥
zoom_matrix/zerosFill!zoom_matrix/zeros/packed:output:0 zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros~
zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/zeros_1/packed/1¹
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
zoom_matrix/zeros_1/Const­
zoom_matrix/zeros_1Fill#zoom_matrix/zeros_1/packed:output:0"zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros_1
!zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2#
!zoom_matrix/strided_slice_4/stack
#zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2%
#zoom_matrix/strided_slice_4/stack_1
#zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2%
#zoom_matrix/strided_slice_4/stack_2ñ
zoom_matrix/strided_slice_4StridedSliceconcat:output:0*zoom_matrix/strided_slice_4/stack:output:0,zoom_matrix/strided_slice_4/stack_1:output:0,zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
zoom_matrix/zeros_2/packed/1¹
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
zoom_matrix/zeros_2/Const­
zoom_matrix/zeros_2Fill#zoom_matrix/zeros_2/packed:output:0"zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/zeros_2t
zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
zoom_matrix/concat/axisá
zoom_matrix/concatConcatV2$zoom_matrix/strided_slice_3:output:0zoom_matrix/zeros:output:0zoom_matrix/mul:z:0zoom_matrix/zeros_1:output:0$zoom_matrix/strided_slice_4:output:0zoom_matrix/mul_1:z:0zoom_matrix/zeros_2:output:0 zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
zoom_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
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
transform/fill_valueÅ
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputszoom_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿôô: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
Û
N
2__inference_max_pooling2d_144_layer_call_fn_168965

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_1689592
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
ÿ
F__inference_conv2d_144_layer_call_and_return_conditional_losses_169018

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿôô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
à
G
+__inference_dropout_37_layer_call_fn_170268

inputs
identityÌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1690662
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs

ù
D__inference_dense_96_layer_call_and_return_conditional_losses_170295

inputs3
matmul_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relun
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
b
F__inference_flatten_48_layer_call_and_return_conditional_losses_169074

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ Á 2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs
¦
 
+__inference_conv2d_144_layer_call_fn_170206

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_144_layer_call_and_return_conditional_losses_1690182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿôô: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ô

)__inference_dense_97_layer_call_fn_170323

inputs
unknown:	

	unknown_0:

identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_1691032
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
K
/__inference_random_zoom_40_layer_call_fn_170743

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1685712
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ª
d
H__inference_rescaling_51_layer_call_and_return_conditional_losses_170181

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
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
:ÿÿÿÿÿÿÿÿÿôô2
mulk
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs

è
I__inference_sequential_88_layer_call_and_return_conditional_losses_168953
random_flip_41_input#
random_flip_41_168943:	'
random_rotation_40_168946:	#
random_zoom_40_168949:	
identity¢&random_flip_41/StatefulPartitionedCall¢*random_rotation_40/StatefulPartitionedCall¢&random_zoom_40/StatefulPartitionedCall®
&random_flip_41/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_41_inputrandom_flip_41_168943*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1688882(
&random_flip_41/StatefulPartitionedCallÙ
*random_rotation_40/StatefulPartitionedCallStatefulPartitionedCall/random_flip_41/StatefulPartitionedCall:output:0random_rotation_40_168946*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1688182,
*random_rotation_40/StatefulPartitionedCallÍ
&random_zoom_40/StatefulPartitionedCallStatefulPartitionedCall3random_rotation_40/StatefulPartitionedCall:output:0random_zoom_40_168949*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1686872(
&random_zoom_40/StatefulPartitionedCall
IdentityIdentity/random_zoom_40/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

IdentityÍ
NoOpNoOp'^random_flip_41/StatefulPartitionedCall+^random_rotation_40/StatefulPartitionedCall'^random_zoom_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿôô: : : 2P
&random_flip_41/StatefulPartitionedCall&random_flip_41/StatefulPartitionedCall2X
*random_rotation_40/StatefulPartitionedCall*random_rotation_40/StatefulPartitionedCall2P
&random_zoom_40/StatefulPartitionedCall&random_zoom_40/StatefulPartitionedCall:g c
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
.
_user_specified_namerandom_flip_41_input
Û
N
2__inference_max_pooling2d_146_layer_call_fn_168989

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_1689832
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
I
-__inference_rescaling_51_layer_call_fn_170186

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_1690052
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
äX

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

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename´
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*Æ
value¼B¹+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-2/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÞ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:+*
dtype0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesé
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_144_kernel_read_readvariableop*savev2_conv2d_144_bias_read_readvariableop,savev2_conv2d_145_kernel_read_readvariableop*savev2_conv2d_145_bias_read_readvariableop,savev2_conv2d_146_kernel_read_readvariableop*savev2_conv2d_146_bias_read_readvariableop*savev2_dense_96_kernel_read_readvariableop(savev2_dense_96_bias_read_readvariableop*savev2_dense_97_kernel_read_readvariableop(savev2_dense_97_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop%savev2_variable_2_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_144_kernel_m_read_readvariableop1savev2_adam_conv2d_144_bias_m_read_readvariableop3savev2_adam_conv2d_145_kernel_m_read_readvariableop1savev2_adam_conv2d_145_bias_m_read_readvariableop3savev2_adam_conv2d_146_kernel_m_read_readvariableop1savev2_adam_conv2d_146_bias_m_read_readvariableop1savev2_adam_dense_96_kernel_m_read_readvariableop/savev2_adam_dense_96_bias_m_read_readvariableop1savev2_adam_dense_97_kernel_m_read_readvariableop/savev2_adam_dense_97_bias_m_read_readvariableop3savev2_adam_conv2d_144_kernel_v_read_readvariableop1savev2_adam_conv2d_144_bias_v_read_readvariableop3savev2_adam_conv2d_145_kernel_v_read_readvariableop1savev2_adam_conv2d_145_bias_v_read_readvariableop3savev2_adam_conv2d_146_kernel_v_read_readvariableop1savev2_adam_conv2d_146_bias_v_read_readvariableop1savev2_adam_dense_96_kernel_v_read_readvariableop/savev2_adam_dense_96_bias_v_read_readvariableop1savev2_adam_dense_97_kernel_v_read_readvariableop/savev2_adam_dense_97_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+				2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapesò
ï: ::: : : @:@:::	
:
: : : : : :::: : : : ::: : : @:@:::	
:
::: : : @:@:::	
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
::!

_output_shapes	
::%	!

_output_shapes
:	
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
::!

_output_shapes	
::%!

_output_shapes
:	
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
::!(

_output_shapes	
::%)!

_output_shapes
:	
: *

_output_shapes
:
:+

_output_shapes
: 

 
+__inference_conv2d_146_layer_call_fn_170246

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_146_layer_call_and_return_conditional_losses_1690542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ}} : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}} 
 
_user_specified_nameinputs
Ô
G
+__inference_flatten_48_layer_call_fn_170284

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_48_layer_call_and_return_conditional_losses_1690742
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs
¤
f
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170416

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
Ã
d
+__inference_dropout_37_layer_call_fn_170273

inputs
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1691792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs
Õ8
É
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
dense_96_169276:
dense_96_169278:	"
dense_97_169281:	

dense_97_169283:

identity¢"conv2d_144/StatefulPartitionedCall¢"conv2d_145/StatefulPartitionedCall¢"conv2d_146/StatefulPartitionedCall¢ dense_96/StatefulPartitionedCall¢ dense_97/StatefulPartitionedCall¢"dropout_37/StatefulPartitionedCall¢%sequential_88/StatefulPartitionedCallÊ
%sequential_88/StatefulPartitionedCallStatefulPartitionedCallinputssequential_88_169248sequential_88_169250sequential_88_169252*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1689132'
%sequential_88/StatefulPartitionedCall
rescaling_51/PartitionedCallPartitionedCall.sequential_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_1690052
rescaling_51/PartitionedCallÇ
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall%rescaling_51/PartitionedCall:output:0conv2d_144_169256conv2d_144_169258*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_144_layer_call_and_return_conditional_losses_1690182$
"conv2d_144/StatefulPartitionedCall
!max_pooling2d_144/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_1689592#
!max_pooling2d_144/PartitionedCallÌ
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_144/PartitionedCall:output:0conv2d_145_169262conv2d_145_169264*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_145_layer_call_and_return_conditional_losses_1690362$
"conv2d_145/StatefulPartitionedCall
!max_pooling2d_145/PartitionedCallPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}} * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_1689712#
!max_pooling2d_145/PartitionedCallÊ
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_145/PartitionedCall:output:0conv2d_146_169268conv2d_146_169270*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_146_layer_call_and_return_conditional_losses_1690542$
"conv2d_146/StatefulPartitionedCall
!max_pooling2d_146/PartitionedCallPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_1689832#
!max_pooling2d_146/PartitionedCall
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1691792$
"dropout_37/StatefulPartitionedCall
flatten_48/PartitionedCallPartitionedCall+dropout_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_48_layer_call_and_return_conditional_losses_1690742
flatten_48/PartitionedCall²
 dense_96/StatefulPartitionedCallStatefulPartitionedCall#flatten_48/PartitionedCall:output:0dense_96_169276dense_96_169278*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_1690872"
 dense_96/StatefulPartitionedCall·
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_169281dense_97_169283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_1691032"
 dense_97/StatefulPartitionedCall
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

IdentityÐ
NoOpNoOp#^conv2d_144/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall&^sequential_88/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : 2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2N
%sequential_88/StatefulPartitionedCall%sequential_88/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
Ø
æ
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
'dense_96_matmul_readvariableop_resource:7
(dense_96_biasadd_readvariableop_resource:	:
'dense_97_matmul_readvariableop_resource:	
6
(dense_97_biasadd_readvariableop_resource:

identity¢!conv2d_144/BiasAdd/ReadVariableOp¢ conv2d_144/Conv2D/ReadVariableOp¢!conv2d_145/BiasAdd/ReadVariableOp¢ conv2d_145/Conv2D/ReadVariableOp¢!conv2d_146/BiasAdd/ReadVariableOp¢ conv2d_146/Conv2D/ReadVariableOp¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOp¢Esequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkip¢lsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢ssequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢@sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip¢<sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkipÆ
<sequential_88/random_flip_41/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_88/random_flip_41/stateful_uniform_full_int/shapeÆ
<sequential_88/random_flip_41/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_88/random_flip_41/stateful_uniform_full_int/Const±
;sequential_88/random_flip_41/stateful_uniform_full_int/ProdProdEsequential_88/random_flip_41/stateful_uniform_full_int/shape:output:0Esequential_88/random_flip_41/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2=
;sequential_88/random_flip_41/stateful_uniform_full_int/ProdÀ
=sequential_88/random_flip_41/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_88/random_flip_41/stateful_uniform_full_int/Cast/xü
=sequential_88/random_flip_41/stateful_uniform_full_int/Cast_1CastDsequential_88/random_flip_41/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2?
=sequential_88/random_flip_41/stateful_uniform_full_int/Cast_1
Esequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipNsequential_88_random_flip_41_stateful_uniform_full_int_rngreadandskip_resourceFsequential_88/random_flip_41/stateful_uniform_full_int/Cast/x:output:0Asequential_88/random_flip_41/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2G
Esequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkipâ
Jsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2L
Jsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stackæ
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_1æ
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_2²
Dsequential_88/random_flip_41/stateful_uniform_full_int/strided_sliceStridedSliceMsequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkip:value:0Ssequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack:output:0Usequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_1:output:0Usequential_88/random_flip_41/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2F
Dsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice
>sequential_88/random_flip_41/stateful_uniform_full_int/BitcastBitcastMsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02@
>sequential_88/random_flip_41/stateful_uniform_full_int/Bitcastæ
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stackê
Nsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2P
Nsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1ê
Nsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2P
Nsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2ª
Fsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1StridedSliceMsequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkip:value:0Usequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack:output:0Wsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Wsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2H
Fsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1
@sequential_88/random_flip_41/stateful_uniform_full_int/Bitcast_1BitcastOsequential_88/random_flip_41/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02B
@sequential_88/random_flip_41/stateful_uniform_full_int/Bitcast_1º
:sequential_88/random_flip_41/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_88/random_flip_41/stateful_uniform_full_int/algÜ
6sequential_88/random_flip_41/stateful_uniform_full_intStatelessRandomUniformFullIntV2Esequential_88/random_flip_41/stateful_uniform_full_int/shape:output:0Isequential_88/random_flip_41/stateful_uniform_full_int/Bitcast_1:output:0Gsequential_88/random_flip_41/stateful_uniform_full_int/Bitcast:output:0Csequential_88/random_flip_41/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	28
6sequential_88/random_flip_41/stateful_uniform_full_int
'sequential_88/random_flip_41/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2)
'sequential_88/random_flip_41/zeros_likeõ
"sequential_88/random_flip_41/stackPack?sequential_88/random_flip_41/stateful_uniform_full_int:output:00sequential_88/random_flip_41/zeros_like:output:0*
N*
T0	*
_output_shapes

:2$
"sequential_88/random_flip_41/stackµ
0sequential_88/random_flip_41/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0sequential_88/random_flip_41/strided_slice/stack¹
2sequential_88/random_flip_41/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2sequential_88/random_flip_41/strided_slice/stack_1¹
2sequential_88/random_flip_41/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2sequential_88/random_flip_41/strided_slice/stack_2¶
*sequential_88/random_flip_41/strided_sliceStridedSlice+sequential_88/random_flip_41/stack:output:09sequential_88/random_flip_41/strided_slice/stack:output:0;sequential_88/random_flip_41/strided_slice/stack_1:output:0;sequential_88/random_flip_41/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2,
*sequential_88/random_flip_41/strided_slice
Psequential_88/random_flip_41/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2R
Psequential_88/random_flip_41/stateless_random_flip_left_right/control_dependency
Csequential_88/random_flip_41/stateless_random_flip_left_right/ShapeShapeYsequential_88/random_flip_41/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2E
Csequential_88/random_flip_41/stateless_random_flip_left_right/Shapeð
Qsequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
Qsequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stackô
Ssequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_1ô
Ssequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
Ssequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_2Ö
Ksequential_88/random_flip_41/stateless_random_flip_left_right/strided_sliceStridedSliceLsequential_88/random_flip_41/stateless_random_flip_left_right/Shape:output:0Zsequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack:output:0\sequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_1:output:0\sequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2M
Ksequential_88/random_flip_41/stateless_random_flip_left_right/strided_sliceÈ
\sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shapePackTsequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2^
\sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shapeý
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2\
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/miný
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2\
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/maxá
ssequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter3sequential_88/random_flip_41/strided_slice:output:0* 
_output_shapes
::2u
ssequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter
lsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgt^sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2n
lsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgø
osequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2esequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0ysequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0}sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0rsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2q
osequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2ª
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/subSubcsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/max:output:0csequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2\
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/subÇ
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mulMulxsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0^sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2\
Zsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mul¬
Vsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniformAddV2^sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0csequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
Vsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniformà
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2O
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/1à
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2O
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/2à
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2O
Msequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/3®
Ksequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shapePackTsequential_88/random_flip_41/stateless_random_flip_left_right/strided_slice:output:0Vsequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/1:output:0Vsequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/2:output:0Vsequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2M
Ksequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape
Esequential_88/random_flip_41/stateless_random_flip_left_right/ReshapeReshapeZsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform:z:0Tsequential_88/random_flip_41/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2G
Esequential_88/random_flip_41/stateless_random_flip_left_right/Reshape
Csequential_88/random_flip_41/stateless_random_flip_left_right/RoundRoundNsequential_88/random_flip_41/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
Csequential_88/random_flip_41/stateless_random_flip_left_right/Roundæ
Lsequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2/axis
Gsequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2	ReverseV2Ysequential_88/random_flip_41/stateless_random_flip_left_right/control_dependency:output:0Usequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2I
Gsequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2ä
Asequential_88/random_flip_41/stateless_random_flip_left_right/mulMulGsequential_88/random_flip_41/stateless_random_flip_left_right/Round:y:0Psequential_88/random_flip_41/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2C
Asequential_88/random_flip_41/stateless_random_flip_left_right/mulÏ
Csequential_88/random_flip_41/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2E
Csequential_88/random_flip_41/stateless_random_flip_left_right/sub/xÞ
Asequential_88/random_flip_41/stateless_random_flip_left_right/subSubLsequential_88/random_flip_41/stateless_random_flip_left_right/sub/x:output:0Gsequential_88/random_flip_41/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
Asequential_88/random_flip_41/stateless_random_flip_left_right/subï
Csequential_88/random_flip_41/stateless_random_flip_left_right/mul_1MulEsequential_88/random_flip_41/stateless_random_flip_left_right/sub:z:0Ysequential_88/random_flip_41/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2E
Csequential_88/random_flip_41/stateless_random_flip_left_right/mul_1Û
Asequential_88/random_flip_41/stateless_random_flip_left_right/addAddV2Esequential_88/random_flip_41/stateless_random_flip_left_right/mul:z:0Gsequential_88/random_flip_41/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2C
Asequential_88/random_flip_41/stateless_random_flip_left_right/addÅ
&sequential_88/random_rotation_40/ShapeShapeEsequential_88/random_flip_41/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2(
&sequential_88/random_rotation_40/Shape¶
4sequential_88/random_rotation_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4sequential_88/random_rotation_40/strided_slice/stackº
6sequential_88/random_rotation_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_88/random_rotation_40/strided_slice/stack_1º
6sequential_88/random_rotation_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_88/random_rotation_40/strided_slice/stack_2¨
.sequential_88/random_rotation_40/strided_sliceStridedSlice/sequential_88/random_rotation_40/Shape:output:0=sequential_88/random_rotation_40/strided_slice/stack:output:0?sequential_88/random_rotation_40/strided_slice/stack_1:output:0?sequential_88/random_rotation_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_88/random_rotation_40/strided_sliceº
6sequential_88/random_rotation_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6sequential_88/random_rotation_40/strided_slice_1/stack¾
8sequential_88/random_rotation_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_88/random_rotation_40/strided_slice_1/stack_1¾
8sequential_88/random_rotation_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_88/random_rotation_40/strided_slice_1/stack_2²
0sequential_88/random_rotation_40/strided_slice_1StridedSlice/sequential_88/random_rotation_40/Shape:output:0?sequential_88/random_rotation_40/strided_slice_1/stack:output:0Asequential_88/random_rotation_40/strided_slice_1/stack_1:output:0Asequential_88/random_rotation_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_88/random_rotation_40/strided_slice_1Á
%sequential_88/random_rotation_40/CastCast9sequential_88/random_rotation_40/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%sequential_88/random_rotation_40/Castº
6sequential_88/random_rotation_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6sequential_88/random_rotation_40/strided_slice_2/stack¾
8sequential_88/random_rotation_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_88/random_rotation_40/strided_slice_2/stack_1¾
8sequential_88/random_rotation_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_88/random_rotation_40/strided_slice_2/stack_2²
0sequential_88/random_rotation_40/strided_slice_2StridedSlice/sequential_88/random_rotation_40/Shape:output:0?sequential_88/random_rotation_40/strided_slice_2/stack:output:0Asequential_88/random_rotation_40/strided_slice_2/stack_1:output:0Asequential_88/random_rotation_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_88/random_rotation_40/strided_slice_2Å
'sequential_88/random_rotation_40/Cast_1Cast9sequential_88/random_rotation_40/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'sequential_88/random_rotation_40/Cast_1á
7sequential_88/random_rotation_40/stateful_uniform/shapePack7sequential_88/random_rotation_40/strided_slice:output:0*
N*
T0*
_output_shapes
:29
7sequential_88/random_rotation_40/stateful_uniform/shape³
5sequential_88/random_rotation_40/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5sequential_88/random_rotation_40/stateful_uniform/min³
5sequential_88/random_rotation_40/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5sequential_88/random_rotation_40/stateful_uniform/max¼
7sequential_88/random_rotation_40/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_88/random_rotation_40/stateful_uniform/Const
6sequential_88/random_rotation_40/stateful_uniform/ProdProd@sequential_88/random_rotation_40/stateful_uniform/shape:output:0@sequential_88/random_rotation_40/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/stateful_uniform/Prod¶
8sequential_88/random_rotation_40/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_88/random_rotation_40/stateful_uniform/Cast/xí
8sequential_88/random_rotation_40/stateful_uniform/Cast_1Cast?sequential_88/random_rotation_40/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2:
8sequential_88/random_rotation_40/stateful_uniform/Cast_1þ
@sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkipRngReadAndSkipIsequential_88_random_rotation_40_stateful_uniform_rngreadandskip_resourceAsequential_88/random_rotation_40/stateful_uniform/Cast/x:output:0<sequential_88/random_rotation_40/stateful_uniform/Cast_1:y:0*
_output_shapes
:2B
@sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkipØ
Esequential_88/random_rotation_40/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_88/random_rotation_40/stateful_uniform/strided_slice/stackÜ
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_1Ü
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_2
?sequential_88/random_rotation_40/stateful_uniform/strided_sliceStridedSliceHsequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip:value:0Nsequential_88/random_rotation_40/stateful_uniform/strided_slice/stack:output:0Psequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_1:output:0Psequential_88/random_rotation_40/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2A
?sequential_88/random_rotation_40/stateful_uniform/strided_sliceü
9sequential_88/random_rotation_40/stateful_uniform/BitcastBitcastHsequential_88/random_rotation_40/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02;
9sequential_88/random_rotation_40/stateful_uniform/BitcastÜ
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stackà
Isequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_1à
Isequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2K
Isequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_2
Asequential_88/random_rotation_40/stateful_uniform/strided_slice_1StridedSliceHsequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip:value:0Psequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack:output:0Rsequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_1:output:0Rsequential_88/random_rotation_40/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2C
Asequential_88/random_rotation_40/stateful_uniform/strided_slice_1
;sequential_88/random_rotation_40/stateful_uniform/Bitcast_1BitcastJsequential_88/random_rotation_40/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02=
;sequential_88/random_rotation_40/stateful_uniform/Bitcast_1â
Nsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2P
Nsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2/algþ
Jsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2@sequential_88/random_rotation_40/stateful_uniform/shape:output:0Dsequential_88/random_rotation_40/stateful_uniform/Bitcast_1:output:0Bsequential_88/random_rotation_40/stateful_uniform/Bitcast:output:0Wsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2L
Jsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2
5sequential_88/random_rotation_40/stateful_uniform/subSub>sequential_88/random_rotation_40/stateful_uniform/max:output:0>sequential_88/random_rotation_40/stateful_uniform/min:output:0*
T0*
_output_shapes
: 27
5sequential_88/random_rotation_40/stateful_uniform/sub³
5sequential_88/random_rotation_40/stateful_uniform/mulMulSsequential_88/random_rotation_40/stateful_uniform/StatelessRandomUniformV2:output:09sequential_88/random_rotation_40/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5sequential_88/random_rotation_40/stateful_uniform/mul
1sequential_88/random_rotation_40/stateful_uniformAddV29sequential_88/random_rotation_40/stateful_uniform/mul:z:0>sequential_88/random_rotation_40/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_88/random_rotation_40/stateful_uniformµ
6sequential_88/random_rotation_40/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?28
6sequential_88/random_rotation_40/rotation_matrix/sub/y
4sequential_88/random_rotation_40/rotation_matrix/subSub+sequential_88/random_rotation_40/Cast_1:y:0?sequential_88/random_rotation_40/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 26
4sequential_88/random_rotation_40/rotation_matrix/subØ
4sequential_88/random_rotation_40/rotation_matrix/CosCos5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4sequential_88/random_rotation_40/rotation_matrix/Cos¹
8sequential_88/random_rotation_40/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8sequential_88/random_rotation_40/rotation_matrix/sub_1/y
6sequential_88/random_rotation_40/rotation_matrix/sub_1Sub+sequential_88/random_rotation_40/Cast_1:y:0Asequential_88/random_rotation_40/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/rotation_matrix/sub_1
4sequential_88/random_rotation_40/rotation_matrix/mulMul8sequential_88/random_rotation_40/rotation_matrix/Cos:y:0:sequential_88/random_rotation_40/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4sequential_88/random_rotation_40/rotation_matrix/mulØ
4sequential_88/random_rotation_40/rotation_matrix/SinSin5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4sequential_88/random_rotation_40/rotation_matrix/Sin¹
8sequential_88/random_rotation_40/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8sequential_88/random_rotation_40/rotation_matrix/sub_2/y
6sequential_88/random_rotation_40/rotation_matrix/sub_2Sub)sequential_88/random_rotation_40/Cast:y:0Asequential_88/random_rotation_40/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/rotation_matrix/sub_2
6sequential_88/random_rotation_40/rotation_matrix/mul_1Mul8sequential_88/random_rotation_40/rotation_matrix/Sin:y:0:sequential_88/random_rotation_40/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/mul_1
6sequential_88/random_rotation_40/rotation_matrix/sub_3Sub8sequential_88/random_rotation_40/rotation_matrix/mul:z:0:sequential_88/random_rotation_40/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/sub_3
6sequential_88/random_rotation_40/rotation_matrix/sub_4Sub8sequential_88/random_rotation_40/rotation_matrix/sub:z:0:sequential_88/random_rotation_40/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/sub_4½
:sequential_88/random_rotation_40/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:sequential_88/random_rotation_40/rotation_matrix/truediv/y®
8sequential_88/random_rotation_40/rotation_matrix/truedivRealDiv:sequential_88/random_rotation_40/rotation_matrix/sub_4:z:0Csequential_88/random_rotation_40/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8sequential_88/random_rotation_40/rotation_matrix/truediv¹
8sequential_88/random_rotation_40/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8sequential_88/random_rotation_40/rotation_matrix/sub_5/y
6sequential_88/random_rotation_40/rotation_matrix/sub_5Sub)sequential_88/random_rotation_40/Cast:y:0Asequential_88/random_rotation_40/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/rotation_matrix/sub_5Ü
6sequential_88/random_rotation_40/rotation_matrix/Sin_1Sin5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/Sin_1¹
8sequential_88/random_rotation_40/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8sequential_88/random_rotation_40/rotation_matrix/sub_6/y
6sequential_88/random_rotation_40/rotation_matrix/sub_6Sub+sequential_88/random_rotation_40/Cast_1:y:0Asequential_88/random_rotation_40/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/rotation_matrix/sub_6
6sequential_88/random_rotation_40/rotation_matrix/mul_2Mul:sequential_88/random_rotation_40/rotation_matrix/Sin_1:y:0:sequential_88/random_rotation_40/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/mul_2Ü
6sequential_88/random_rotation_40/rotation_matrix/Cos_1Cos5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/Cos_1¹
8sequential_88/random_rotation_40/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8sequential_88/random_rotation_40/rotation_matrix/sub_7/y
6sequential_88/random_rotation_40/rotation_matrix/sub_7Sub)sequential_88/random_rotation_40/Cast:y:0Asequential_88/random_rotation_40/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 28
6sequential_88/random_rotation_40/rotation_matrix/sub_7
6sequential_88/random_rotation_40/rotation_matrix/mul_3Mul:sequential_88/random_rotation_40/rotation_matrix/Cos_1:y:0:sequential_88/random_rotation_40/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/mul_3
4sequential_88/random_rotation_40/rotation_matrix/addAddV2:sequential_88/random_rotation_40/rotation_matrix/mul_2:z:0:sequential_88/random_rotation_40/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4sequential_88/random_rotation_40/rotation_matrix/add
6sequential_88/random_rotation_40/rotation_matrix/sub_8Sub:sequential_88/random_rotation_40/rotation_matrix/sub_5:z:08sequential_88/random_rotation_40/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/sub_8Á
<sequential_88/random_rotation_40/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2>
<sequential_88/random_rotation_40/rotation_matrix/truediv_1/y´
:sequential_88/random_rotation_40/rotation_matrix/truediv_1RealDiv:sequential_88/random_rotation_40/rotation_matrix/sub_8:z:0Esequential_88/random_rotation_40/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:sequential_88/random_rotation_40/rotation_matrix/truediv_1Õ
6sequential_88/random_rotation_40/rotation_matrix/ShapeShape5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*
_output_shapes
:28
6sequential_88/random_rotation_40/rotation_matrix/ShapeÖ
Dsequential_88/random_rotation_40/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2F
Dsequential_88/random_rotation_40/rotation_matrix/strided_slice/stackÚ
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_1Ú
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_2
>sequential_88/random_rotation_40/rotation_matrix/strided_sliceStridedSlice?sequential_88/random_rotation_40/rotation_matrix/Shape:output:0Msequential_88/random_rotation_40/rotation_matrix/strided_slice/stack:output:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_1:output:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2@
>sequential_88/random_rotation_40/rotation_matrix/strided_sliceÜ
6sequential_88/random_rotation_40/rotation_matrix/Cos_2Cos5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/Cos_2á
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stackå
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_1å
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_2½
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_1StridedSlice:sequential_88/random_rotation_40/rotation_matrix/Cos_2:y:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_1Ü
6sequential_88/random_rotation_40/rotation_matrix/Sin_2Sin5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/Sin_2á
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stackå
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_1å
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_2½
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_2StridedSlice:sequential_88/random_rotation_40/rotation_matrix/Sin_2:y:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_2ð
4sequential_88/random_rotation_40/rotation_matrix/NegNegIsequential_88/random_rotation_40/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4sequential_88/random_rotation_40/rotation_matrix/Negá
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stackå
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_1å
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_2¿
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_3StridedSlice<sequential_88/random_rotation_40/rotation_matrix/truediv:z:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_3Ü
6sequential_88/random_rotation_40/rotation_matrix/Sin_3Sin5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/Sin_3á
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stackå
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_1å
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_2½
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_4StridedSlice:sequential_88/random_rotation_40/rotation_matrix/Sin_3:y:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_4Ü
6sequential_88/random_rotation_40/rotation_matrix/Cos_3Cos5sequential_88/random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/Cos_3á
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stackå
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_1å
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_2½
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_5StridedSlice:sequential_88/random_rotation_40/rotation_matrix/Cos_3:y:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_5á
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stackå
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_1å
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2J
Hsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_2Á
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_6StridedSlice>sequential_88/random_rotation_40/rotation_matrix/truediv_1:z:0Osequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_1:output:0Qsequential_88/random_rotation_40/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2B
@sequential_88/random_rotation_40/rotation_matrix/strided_slice_6Ä
?sequential_88/random_rotation_40/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2A
?sequential_88/random_rotation_40/rotation_matrix/zeros/packed/1Ç
=sequential_88/random_rotation_40/rotation_matrix/zeros/packedPackGsequential_88/random_rotation_40/rotation_matrix/strided_slice:output:0Hsequential_88/random_rotation_40/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2?
=sequential_88/random_rotation_40/rotation_matrix/zeros/packedÁ
<sequential_88/random_rotation_40/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2>
<sequential_88/random_rotation_40/rotation_matrix/zeros/Const¹
6sequential_88/random_rotation_40/rotation_matrix/zerosFillFsequential_88/random_rotation_40/rotation_matrix/zeros/packed:output:0Esequential_88/random_rotation_40/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_88/random_rotation_40/rotation_matrix/zeros¾
<sequential_88/random_rotation_40/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2>
<sequential_88/random_rotation_40/rotation_matrix/concat/axisò
7sequential_88/random_rotation_40/rotation_matrix/concatConcatV2Isequential_88/random_rotation_40/rotation_matrix/strided_slice_1:output:08sequential_88/random_rotation_40/rotation_matrix/Neg:y:0Isequential_88/random_rotation_40/rotation_matrix/strided_slice_3:output:0Isequential_88/random_rotation_40/rotation_matrix/strided_slice_4:output:0Isequential_88/random_rotation_40/rotation_matrix/strided_slice_5:output:0Isequential_88/random_rotation_40/rotation_matrix/strided_slice_6:output:0?sequential_88/random_rotation_40/rotation_matrix/zeros:output:0Esequential_88/random_rotation_40/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7sequential_88/random_rotation_40/rotation_matrix/concatÙ
0sequential_88/random_rotation_40/transform/ShapeShapeEsequential_88/random_flip_41/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:22
0sequential_88/random_rotation_40/transform/ShapeÊ
>sequential_88/random_rotation_40/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_88/random_rotation_40/transform/strided_slice/stackÎ
@sequential_88/random_rotation_40/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_88/random_rotation_40/transform/strided_slice/stack_1Î
@sequential_88/random_rotation_40/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_88/random_rotation_40/transform/strided_slice/stack_2Ð
8sequential_88/random_rotation_40/transform/strided_sliceStridedSlice9sequential_88/random_rotation_40/transform/Shape:output:0Gsequential_88/random_rotation_40/transform/strided_slice/stack:output:0Isequential_88/random_rotation_40/transform/strided_slice/stack_1:output:0Isequential_88/random_rotation_40/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2:
8sequential_88/random_rotation_40/transform/strided_slice³
5sequential_88/random_rotation_40/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5sequential_88/random_rotation_40/transform/fill_value­
Esequential_88/random_rotation_40/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Esequential_88/random_flip_41/stateless_random_flip_left_right/add:z:0@sequential_88/random_rotation_40/rotation_matrix/concat:output:0Asequential_88/random_rotation_40/transform/strided_slice:output:0>sequential_88/random_rotation_40/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2G
Esequential_88/random_rotation_40/transform/ImageProjectiveTransformV3Ò
"sequential_88/random_zoom_40/ShapeShapeZsequential_88/random_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2$
"sequential_88/random_zoom_40/Shape®
0sequential_88/random_zoom_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_88/random_zoom_40/strided_slice/stack²
2sequential_88/random_zoom_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_88/random_zoom_40/strided_slice/stack_1²
2sequential_88/random_zoom_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_88/random_zoom_40/strided_slice/stack_2
*sequential_88/random_zoom_40/strided_sliceStridedSlice+sequential_88/random_zoom_40/Shape:output:09sequential_88/random_zoom_40/strided_slice/stack:output:0;sequential_88/random_zoom_40/strided_slice/stack_1:output:0;sequential_88/random_zoom_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_88/random_zoom_40/strided_slice²
2sequential_88/random_zoom_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2sequential_88/random_zoom_40/strided_slice_1/stack¶
4sequential_88/random_zoom_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_88/random_zoom_40/strided_slice_1/stack_1¶
4sequential_88/random_zoom_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_88/random_zoom_40/strided_slice_1/stack_2
,sequential_88/random_zoom_40/strided_slice_1StridedSlice+sequential_88/random_zoom_40/Shape:output:0;sequential_88/random_zoom_40/strided_slice_1/stack:output:0=sequential_88/random_zoom_40/strided_slice_1/stack_1:output:0=sequential_88/random_zoom_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_88/random_zoom_40/strided_slice_1µ
!sequential_88/random_zoom_40/CastCast5sequential_88/random_zoom_40/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!sequential_88/random_zoom_40/Cast²
2sequential_88/random_zoom_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2sequential_88/random_zoom_40/strided_slice_2/stack¶
4sequential_88/random_zoom_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_88/random_zoom_40/strided_slice_2/stack_1¶
4sequential_88/random_zoom_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_88/random_zoom_40/strided_slice_2/stack_2
,sequential_88/random_zoom_40/strided_slice_2StridedSlice+sequential_88/random_zoom_40/Shape:output:0;sequential_88/random_zoom_40/strided_slice_2/stack:output:0=sequential_88/random_zoom_40/strided_slice_2/stack_1:output:0=sequential_88/random_zoom_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_88/random_zoom_40/strided_slice_2¹
#sequential_88/random_zoom_40/Cast_1Cast5sequential_88/random_zoom_40/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#sequential_88/random_zoom_40/Cast_1°
5sequential_88/random_zoom_40/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5sequential_88/random_zoom_40/stateful_uniform/shape/1
3sequential_88/random_zoom_40/stateful_uniform/shapePack3sequential_88/random_zoom_40/strided_slice:output:0>sequential_88/random_zoom_40/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:25
3sequential_88/random_zoom_40/stateful_uniform/shape«
1sequential_88/random_zoom_40/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?23
1sequential_88/random_zoom_40/stateful_uniform/min«
1sequential_88/random_zoom_40/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?23
1sequential_88/random_zoom_40/stateful_uniform/max´
3sequential_88/random_zoom_40/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_88/random_zoom_40/stateful_uniform/Const
2sequential_88/random_zoom_40/stateful_uniform/ProdProd<sequential_88/random_zoom_40/stateful_uniform/shape:output:0<sequential_88/random_zoom_40/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 24
2sequential_88/random_zoom_40/stateful_uniform/Prod®
4sequential_88/random_zoom_40/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_88/random_zoom_40/stateful_uniform/Cast/xá
4sequential_88/random_zoom_40/stateful_uniform/Cast_1Cast;sequential_88/random_zoom_40/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 26
4sequential_88/random_zoom_40/stateful_uniform/Cast_1ê
<sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkipRngReadAndSkipEsequential_88_random_zoom_40_stateful_uniform_rngreadandskip_resource=sequential_88/random_zoom_40/stateful_uniform/Cast/x:output:08sequential_88/random_zoom_40/stateful_uniform/Cast_1:y:0*
_output_shapes
:2>
<sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkipÐ
Asequential_88/random_zoom_40/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_88/random_zoom_40/stateful_uniform/strided_slice/stackÔ
Csequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_1Ô
Csequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_2ü
;sequential_88/random_zoom_40/stateful_uniform/strided_sliceStridedSliceDsequential_88/random_zoom_40/stateful_uniform/RngReadAndSkip:value:0Jsequential_88/random_zoom_40/stateful_uniform/strided_slice/stack:output:0Lsequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_1:output:0Lsequential_88/random_zoom_40/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2=
;sequential_88/random_zoom_40/stateful_uniform/strided_sliceð
5sequential_88/random_zoom_40/stateful_uniform/BitcastBitcastDsequential_88/random_zoom_40/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type027
5sequential_88/random_zoom_40/stateful_uniform/BitcastÔ
Csequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stackØ
Esequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_1Ø
Esequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_2ô
=sequential_88/random_zoom_40/stateful_uniform/strided_slice_1StridedSliceDsequential_88/random_zoom_40/stateful_uniform/RngReadAndSkip:value:0Lsequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack:output:0Nsequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_1:output:0Nsequential_88/random_zoom_40/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2?
=sequential_88/random_zoom_40/stateful_uniform/strided_slice_1ö
7sequential_88/random_zoom_40/stateful_uniform/Bitcast_1BitcastFsequential_88/random_zoom_40/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type029
7sequential_88/random_zoom_40/stateful_uniform/Bitcast_1Ú
Jsequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2L
Jsequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2/algê
Fsequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2<sequential_88/random_zoom_40/stateful_uniform/shape:output:0@sequential_88/random_zoom_40/stateful_uniform/Bitcast_1:output:0>sequential_88/random_zoom_40/stateful_uniform/Bitcast:output:0Ssequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
Fsequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2
1sequential_88/random_zoom_40/stateful_uniform/subSub:sequential_88/random_zoom_40/stateful_uniform/max:output:0:sequential_88/random_zoom_40/stateful_uniform/min:output:0*
T0*
_output_shapes
: 23
1sequential_88/random_zoom_40/stateful_uniform/sub§
1sequential_88/random_zoom_40/stateful_uniform/mulMulOsequential_88/random_zoom_40/stateful_uniform/StatelessRandomUniformV2:output:05sequential_88/random_zoom_40/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_88/random_zoom_40/stateful_uniform/mul
-sequential_88/random_zoom_40/stateful_uniformAddV25sequential_88/random_zoom_40/stateful_uniform/mul:z:0:sequential_88/random_zoom_40/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_88/random_zoom_40/stateful_uniform
(sequential_88/random_zoom_40/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2*
(sequential_88/random_zoom_40/concat/axisª
#sequential_88/random_zoom_40/concatConcatV21sequential_88/random_zoom_40/stateful_uniform:z:01sequential_88/random_zoom_40/stateful_uniform:z:01sequential_88/random_zoom_40/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential_88/random_zoom_40/concat¼
.sequential_88/random_zoom_40/zoom_matrix/ShapeShape,sequential_88/random_zoom_40/concat:output:0*
T0*
_output_shapes
:20
.sequential_88/random_zoom_40/zoom_matrix/ShapeÆ
<sequential_88/random_zoom_40/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_88/random_zoom_40/zoom_matrix/strided_slice/stackÊ
>sequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_1Ê
>sequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_2Ø
6sequential_88/random_zoom_40/zoom_matrix/strided_sliceStridedSlice7sequential_88/random_zoom_40/zoom_matrix/Shape:output:0Esequential_88/random_zoom_40/zoom_matrix/strided_slice/stack:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_1:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6sequential_88/random_zoom_40/zoom_matrix/strided_slice¥
.sequential_88/random_zoom_40/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.sequential_88/random_zoom_40/zoom_matrix/sub/yæ
,sequential_88/random_zoom_40/zoom_matrix/subSub'sequential_88/random_zoom_40/Cast_1:y:07sequential_88/random_zoom_40/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2.
,sequential_88/random_zoom_40/zoom_matrix/sub­
2sequential_88/random_zoom_40/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @24
2sequential_88/random_zoom_40/zoom_matrix/truediv/yÿ
0sequential_88/random_zoom_40/zoom_matrix/truedivRealDiv0sequential_88/random_zoom_40/zoom_matrix/sub:z:0;sequential_88/random_zoom_40/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 22
0sequential_88/random_zoom_40/zoom_matrix/truedivÕ
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stackÙ
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_1Ù
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_2
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_1StridedSlice,sequential_88/random_zoom_40/concat:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_1:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2:
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_1©
0sequential_88/random_zoom_40/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_88/random_zoom_40/zoom_matrix/sub_1/x
.sequential_88/random_zoom_40/zoom_matrix/sub_1Sub9sequential_88/random_zoom_40/zoom_matrix/sub_1/x:output:0Asequential_88/random_zoom_40/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_88/random_zoom_40/zoom_matrix/sub_1ÿ
,sequential_88/random_zoom_40/zoom_matrix/mulMul4sequential_88/random_zoom_40/zoom_matrix/truediv:z:02sequential_88/random_zoom_40/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_88/random_zoom_40/zoom_matrix/mul©
0sequential_88/random_zoom_40/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_88/random_zoom_40/zoom_matrix/sub_2/yê
.sequential_88/random_zoom_40/zoom_matrix/sub_2Sub%sequential_88/random_zoom_40/Cast:y:09sequential_88/random_zoom_40/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 20
.sequential_88/random_zoom_40/zoom_matrix/sub_2±
4sequential_88/random_zoom_40/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @26
4sequential_88/random_zoom_40/zoom_matrix/truediv_1/y
2sequential_88/random_zoom_40/zoom_matrix/truediv_1RealDiv2sequential_88/random_zoom_40/zoom_matrix/sub_2:z:0=sequential_88/random_zoom_40/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 24
2sequential_88/random_zoom_40/zoom_matrix/truediv_1Õ
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stackÙ
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_1Ù
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_2
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_2StridedSlice,sequential_88/random_zoom_40/concat:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_1:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2:
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_2©
0sequential_88/random_zoom_40/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?22
0sequential_88/random_zoom_40/zoom_matrix/sub_3/x
.sequential_88/random_zoom_40/zoom_matrix/sub_3Sub9sequential_88/random_zoom_40/zoom_matrix/sub_3/x:output:0Asequential_88/random_zoom_40/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_88/random_zoom_40/zoom_matrix/sub_3
.sequential_88/random_zoom_40/zoom_matrix/mul_1Mul6sequential_88/random_zoom_40/zoom_matrix/truediv_1:z:02sequential_88/random_zoom_40/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_88/random_zoom_40/zoom_matrix/mul_1Õ
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stackÙ
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_1Ù
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_2
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_3StridedSlice,sequential_88/random_zoom_40/concat:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_1:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2:
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_3´
7sequential_88/random_zoom_40/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :29
7sequential_88/random_zoom_40/zoom_matrix/zeros/packed/1§
5sequential_88/random_zoom_40/zoom_matrix/zeros/packedPack?sequential_88/random_zoom_40/zoom_matrix/strided_slice:output:0@sequential_88/random_zoom_40/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:27
5sequential_88/random_zoom_40/zoom_matrix/zeros/packed±
4sequential_88/random_zoom_40/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4sequential_88/random_zoom_40/zoom_matrix/zeros/Const
.sequential_88/random_zoom_40/zoom_matrix/zerosFill>sequential_88/random_zoom_40/zoom_matrix/zeros/packed:output:0=sequential_88/random_zoom_40/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_88/random_zoom_40/zoom_matrix/zeros¸
9sequential_88/random_zoom_40/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2;
9sequential_88/random_zoom_40/zoom_matrix/zeros_1/packed/1­
7sequential_88/random_zoom_40/zoom_matrix/zeros_1/packedPack?sequential_88/random_zoom_40/zoom_matrix/strided_slice:output:0Bsequential_88/random_zoom_40/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:29
7sequential_88/random_zoom_40/zoom_matrix/zeros_1/packedµ
6sequential_88/random_zoom_40/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6sequential_88/random_zoom_40/zoom_matrix/zeros_1/Const¡
0sequential_88/random_zoom_40/zoom_matrix/zeros_1Fill@sequential_88/random_zoom_40/zoom_matrix/zeros_1/packed:output:0?sequential_88/random_zoom_40/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_88/random_zoom_40/zoom_matrix/zeros_1Õ
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2@
>sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stackÙ
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_1Ù
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2B
@sequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_2
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_4StridedSlice,sequential_88/random_zoom_40/concat:output:0Gsequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_1:output:0Isequential_88/random_zoom_40/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2:
8sequential_88/random_zoom_40/zoom_matrix/strided_slice_4¸
9sequential_88/random_zoom_40/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2;
9sequential_88/random_zoom_40/zoom_matrix/zeros_2/packed/1­
7sequential_88/random_zoom_40/zoom_matrix/zeros_2/packedPack?sequential_88/random_zoom_40/zoom_matrix/strided_slice:output:0Bsequential_88/random_zoom_40/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:29
7sequential_88/random_zoom_40/zoom_matrix/zeros_2/packedµ
6sequential_88/random_zoom_40/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    28
6sequential_88/random_zoom_40/zoom_matrix/zeros_2/Const¡
0sequential_88/random_zoom_40/zoom_matrix/zeros_2Fill@sequential_88/random_zoom_40/zoom_matrix/zeros_2/packed:output:0?sequential_88/random_zoom_40/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_88/random_zoom_40/zoom_matrix/zeros_2®
4sequential_88/random_zoom_40/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_88/random_zoom_40/zoom_matrix/concat/axis
/sequential_88/random_zoom_40/zoom_matrix/concatConcatV2Asequential_88/random_zoom_40/zoom_matrix/strided_slice_3:output:07sequential_88/random_zoom_40/zoom_matrix/zeros:output:00sequential_88/random_zoom_40/zoom_matrix/mul:z:09sequential_88/random_zoom_40/zoom_matrix/zeros_1:output:0Asequential_88/random_zoom_40/zoom_matrix/strided_slice_4:output:02sequential_88/random_zoom_40/zoom_matrix/mul_1:z:09sequential_88/random_zoom_40/zoom_matrix/zeros_2:output:0=sequential_88/random_zoom_40/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_88/random_zoom_40/zoom_matrix/concatæ
,sequential_88/random_zoom_40/transform/ShapeShapeZsequential_88/random_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2.
,sequential_88/random_zoom_40/transform/ShapeÂ
:sequential_88/random_zoom_40/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_88/random_zoom_40/transform/strided_slice/stackÆ
<sequential_88/random_zoom_40/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_88/random_zoom_40/transform/strided_slice/stack_1Æ
<sequential_88/random_zoom_40/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_88/random_zoom_40/transform/strided_slice/stack_2¸
4sequential_88/random_zoom_40/transform/strided_sliceStridedSlice5sequential_88/random_zoom_40/transform/Shape:output:0Csequential_88/random_zoom_40/transform/strided_slice/stack:output:0Esequential_88/random_zoom_40/transform/strided_slice/stack_1:output:0Esequential_88/random_zoom_40/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:26
4sequential_88/random_zoom_40/transform/strided_slice«
1sequential_88/random_zoom_40/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1sequential_88/random_zoom_40/transform/fill_valueª
Asequential_88/random_zoom_40/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Zsequential_88/random_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:08sequential_88/random_zoom_40/zoom_matrix/concat:output:0=sequential_88/random_zoom_40/transform/strided_slice:output:0:sequential_88/random_zoom_40/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
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
 *;2
rescaling_51/Cast/xs
rescaling_51/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_51/Cast_1/xÝ
rescaling_51/mulMulVsequential_88/random_zoom_40/transform/ImageProjectiveTransformV3:transformed_images:0rescaling_51/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
rescaling_51/mul
rescaling_51/addAddV2rescaling_51/mul:z:0rescaling_51/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
rescaling_51/add¶
 conv2d_144/Conv2D/ReadVariableOpReadVariableOp)conv2d_144_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_144/Conv2D/ReadVariableOpÔ
conv2d_144/Conv2DConv2Drescaling_51/add:z:0(conv2d_144/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
paddingSAME*
strides
2
conv2d_144/Conv2D­
!conv2d_144/BiasAdd/ReadVariableOpReadVariableOp*conv2d_144_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_144/BiasAdd/ReadVariableOp¶
conv2d_144/BiasAddBiasAddconv2d_144/Conv2D:output:0)conv2d_144/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
conv2d_144/BiasAdd
conv2d_144/ReluReluconv2d_144/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
conv2d_144/ReluÏ
max_pooling2d_144/MaxPoolMaxPoolconv2d_144/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú*
ksize
*
paddingVALID*
strides
2
max_pooling2d_144/MaxPool¶
 conv2d_145/Conv2D/ReadVariableOpReadVariableOp)conv2d_145_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 conv2d_145/Conv2D/ReadVariableOpâ
conv2d_145/Conv2DConv2D"max_pooling2d_144/MaxPool:output:0(conv2d_145/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú *
paddingSAME*
strides
2
conv2d_145/Conv2D­
!conv2d_145/BiasAdd/ReadVariableOpReadVariableOp*conv2d_145_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_145/BiasAdd/ReadVariableOp¶
conv2d_145/BiasAddBiasAddconv2d_145/Conv2D:output:0)conv2d_145/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2
conv2d_145/BiasAdd
conv2d_145/ReluReluconv2d_145/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2
conv2d_145/ReluÍ
max_pooling2d_145/MaxPoolMaxPoolconv2d_145/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}} *
ksize
*
paddingVALID*
strides
2
max_pooling2d_145/MaxPool¶
 conv2d_146/Conv2D/ReadVariableOpReadVariableOp)conv2d_146_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02"
 conv2d_146/Conv2D/ReadVariableOpà
conv2d_146/Conv2DConv2D"max_pooling2d_145/MaxPool:output:0(conv2d_146/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingSAME*
strides
2
conv2d_146/Conv2D­
!conv2d_146/BiasAdd/ReadVariableOpReadVariableOp*conv2d_146_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_146/BiasAdd/ReadVariableOp´
conv2d_146/BiasAddBiasAddconv2d_146/Conv2D:output:0)conv2d_146/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_146/BiasAdd
conv2d_146/ReluReluconv2d_146/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_146/ReluÍ
max_pooling2d_146/MaxPoolMaxPoolconv2d_146/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
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
 *G­?2
dropout_37/dropout/Const¸
dropout_37/dropout/MulMul"max_pooling2d_146/MaxPool:output:0!dropout_37/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout_37/dropout/Mul
dropout_37/dropout/ShapeShape"max_pooling2d_146/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_37/dropout/ShapeÝ
/dropout_37/dropout/random_uniform/RandomUniformRandomUniform!dropout_37/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
dtype021
/dropout_37/dropout/random_uniform/RandomUniform
!dropout_37/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬>2#
!dropout_37/dropout/GreaterEqual/yò
dropout_37/dropout/GreaterEqualGreaterEqual8dropout_37/dropout/random_uniform/RandomUniform:output:0*dropout_37/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2!
dropout_37/dropout/GreaterEqual¨
dropout_37/dropout/CastCast#dropout_37/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout_37/dropout/Cast®
dropout_37/dropout/Mul_1Muldropout_37/dropout/Mul:z:0dropout_37/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout_37/dropout/Mul_1u
flatten_48/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ Á 2
flatten_48/Const 
flatten_48/ReshapeReshapedropout_37/dropout/Mul_1:z:0flatten_48/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_48/Reshape«
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*!
_output_shapes
:*
dtype02 
dense_96/MatMul/ReadVariableOp¤
dense_96/MatMulMatMulflatten_48/Reshape:output:0&dense_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_96/MatMul¨
dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_96/BiasAdd/ReadVariableOp¦
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_96/BiasAddt
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_96/Relu©
dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02 
dense_97/MatMul/ReadVariableOp£
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_97/MatMul§
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_97/BiasAdd/ReadVariableOp¥
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_97/BiasAddt
IdentityIdentitydense_97/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

IdentityØ
NoOpNoOp"^conv2d_144/BiasAdd/ReadVariableOp!^conv2d_144/Conv2D/ReadVariableOp"^conv2d_145/BiasAdd/ReadVariableOp!^conv2d_145/Conv2D/ReadVariableOp"^conv2d_146/BiasAdd/ReadVariableOp!^conv2d_146/Conv2D/ReadVariableOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOpF^sequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkipm^sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgt^sequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterA^sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip=^sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : 2F
!conv2d_144/BiasAdd/ReadVariableOp!conv2d_144/BiasAdd/ReadVariableOp2D
 conv2d_144/Conv2D/ReadVariableOp conv2d_144/Conv2D/ReadVariableOp2F
!conv2d_145/BiasAdd/ReadVariableOp!conv2d_145/BiasAdd/ReadVariableOp2D
 conv2d_145/Conv2D/ReadVariableOp conv2d_145/Conv2D/ReadVariableOp2F
!conv2d_146/BiasAdd/ReadVariableOp!conv2d_146/BiasAdd/ReadVariableOp2D
 conv2d_146/Conv2D/ReadVariableOp conv2d_146/Conv2D/ReadVariableOp2B
dense_96/BiasAdd/ReadVariableOpdense_96/BiasAdd/ReadVariableOp2@
dense_96/MatMul/ReadVariableOpdense_96/MatMul/ReadVariableOp2B
dense_97/BiasAdd/ReadVariableOpdense_97/BiasAdd/ReadVariableOp2@
dense_97/MatMul/ReadVariableOpdense_97/MatMul/ReadVariableOp2
Esequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkipEsequential_88/random_flip_41/stateful_uniform_full_int/RngReadAndSkip2Ü
lsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlglsequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2ê
ssequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterssequential_88/random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2
@sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip@sequential_88/random_rotation_40/stateful_uniform/RngReadAndSkip2|
<sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkip<sequential_88/random_zoom_40/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
áR
¨

!__inference__wrapped_model_168345
sequential_88_inputQ
7sequential_89_conv2d_144_conv2d_readvariableop_resource:F
8sequential_89_conv2d_144_biasadd_readvariableop_resource:Q
7sequential_89_conv2d_145_conv2d_readvariableop_resource: F
8sequential_89_conv2d_145_biasadd_readvariableop_resource: Q
7sequential_89_conv2d_146_conv2d_readvariableop_resource: @F
8sequential_89_conv2d_146_biasadd_readvariableop_resource:@J
5sequential_89_dense_96_matmul_readvariableop_resource:E
6sequential_89_dense_96_biasadd_readvariableop_resource:	H
5sequential_89_dense_97_matmul_readvariableop_resource:	
D
6sequential_89_dense_97_biasadd_readvariableop_resource:

identity¢/sequential_89/conv2d_144/BiasAdd/ReadVariableOp¢.sequential_89/conv2d_144/Conv2D/ReadVariableOp¢/sequential_89/conv2d_145/BiasAdd/ReadVariableOp¢.sequential_89/conv2d_145/Conv2D/ReadVariableOp¢/sequential_89/conv2d_146/BiasAdd/ReadVariableOp¢.sequential_89/conv2d_146/Conv2D/ReadVariableOp¢-sequential_89/dense_96/BiasAdd/ReadVariableOp¢,sequential_89/dense_96/MatMul/ReadVariableOp¢-sequential_89/dense_97/BiasAdd/ReadVariableOp¢,sequential_89/dense_97/MatMul/ReadVariableOp
!sequential_89/rescaling_51/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2#
!sequential_89/rescaling_51/Cast/x
#sequential_89/rescaling_51/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_89/rescaling_51/Cast_1/xÄ
sequential_89/rescaling_51/mulMulsequential_88_input*sequential_89/rescaling_51/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2 
sequential_89/rescaling_51/mul×
sequential_89/rescaling_51/addAddV2"sequential_89/rescaling_51/mul:z:0,sequential_89/rescaling_51/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2 
sequential_89/rescaling_51/addà
.sequential_89/conv2d_144/Conv2D/ReadVariableOpReadVariableOp7sequential_89_conv2d_144_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.sequential_89/conv2d_144/Conv2D/ReadVariableOp
sequential_89/conv2d_144/Conv2DConv2D"sequential_89/rescaling_51/add:z:06sequential_89/conv2d_144/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
paddingSAME*
strides
2!
sequential_89/conv2d_144/Conv2D×
/sequential_89/conv2d_144/BiasAdd/ReadVariableOpReadVariableOp8sequential_89_conv2d_144_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential_89/conv2d_144/BiasAdd/ReadVariableOpî
 sequential_89/conv2d_144/BiasAddBiasAdd(sequential_89/conv2d_144/Conv2D:output:07sequential_89/conv2d_144/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2"
 sequential_89/conv2d_144/BiasAdd­
sequential_89/conv2d_144/ReluRelu)sequential_89/conv2d_144/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
sequential_89/conv2d_144/Reluù
'sequential_89/max_pooling2d_144/MaxPoolMaxPool+sequential_89/conv2d_144/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú*
ksize
*
paddingVALID*
strides
2)
'sequential_89/max_pooling2d_144/MaxPoolà
.sequential_89/conv2d_145/Conv2D/ReadVariableOpReadVariableOp7sequential_89_conv2d_145_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.sequential_89/conv2d_145/Conv2D/ReadVariableOp
sequential_89/conv2d_145/Conv2DConv2D0sequential_89/max_pooling2d_144/MaxPool:output:06sequential_89/conv2d_145/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú *
paddingSAME*
strides
2!
sequential_89/conv2d_145/Conv2D×
/sequential_89/conv2d_145/BiasAdd/ReadVariableOpReadVariableOp8sequential_89_conv2d_145_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/sequential_89/conv2d_145/BiasAdd/ReadVariableOpî
 sequential_89/conv2d_145/BiasAddBiasAdd(sequential_89/conv2d_145/Conv2D:output:07sequential_89/conv2d_145/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2"
 sequential_89/conv2d_145/BiasAdd­
sequential_89/conv2d_145/ReluRelu)sequential_89/conv2d_145/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2
sequential_89/conv2d_145/Relu÷
'sequential_89/max_pooling2d_145/MaxPoolMaxPool+sequential_89/conv2d_145/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}} *
ksize
*
paddingVALID*
strides
2)
'sequential_89/max_pooling2d_145/MaxPoolà
.sequential_89/conv2d_146/Conv2D/ReadVariableOpReadVariableOp7sequential_89_conv2d_146_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.sequential_89/conv2d_146/Conv2D/ReadVariableOp
sequential_89/conv2d_146/Conv2DConv2D0sequential_89/max_pooling2d_145/MaxPool:output:06sequential_89/conv2d_146/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingSAME*
strides
2!
sequential_89/conv2d_146/Conv2D×
/sequential_89/conv2d_146/BiasAdd/ReadVariableOpReadVariableOp8sequential_89_conv2d_146_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/sequential_89/conv2d_146/BiasAdd/ReadVariableOpì
 sequential_89/conv2d_146/BiasAddBiasAdd(sequential_89/conv2d_146/Conv2D:output:07sequential_89/conv2d_146/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2"
 sequential_89/conv2d_146/BiasAdd«
sequential_89/conv2d_146/ReluRelu)sequential_89/conv2d_146/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
sequential_89/conv2d_146/Relu÷
'sequential_89/max_pooling2d_146/MaxPoolMaxPool+sequential_89/conv2d_146/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
ksize
*
paddingVALID*
strides
2)
'sequential_89/max_pooling2d_146/MaxPool¾
!sequential_89/dropout_37/IdentityIdentity0sequential_89/max_pooling2d_146/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2#
!sequential_89/dropout_37/Identity
sequential_89/flatten_48/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ Á 2 
sequential_89/flatten_48/ConstØ
 sequential_89/flatten_48/ReshapeReshape*sequential_89/dropout_37/Identity:output:0'sequential_89/flatten_48/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_89/flatten_48/ReshapeÕ
,sequential_89/dense_96/MatMul/ReadVariableOpReadVariableOp5sequential_89_dense_96_matmul_readvariableop_resource*!
_output_shapes
:*
dtype02.
,sequential_89/dense_96/MatMul/ReadVariableOpÜ
sequential_89/dense_96/MatMulMatMul)sequential_89/flatten_48/Reshape:output:04sequential_89/dense_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_89/dense_96/MatMulÒ
-sequential_89/dense_96/BiasAdd/ReadVariableOpReadVariableOp6sequential_89_dense_96_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02/
-sequential_89/dense_96/BiasAdd/ReadVariableOpÞ
sequential_89/dense_96/BiasAddBiasAdd'sequential_89/dense_96/MatMul:product:05sequential_89/dense_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_89/dense_96/BiasAdd
sequential_89/dense_96/ReluRelu'sequential_89/dense_96/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_89/dense_96/ReluÓ
,sequential_89/dense_97/MatMul/ReadVariableOpReadVariableOp5sequential_89_dense_97_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02.
,sequential_89/dense_97/MatMul/ReadVariableOpÛ
sequential_89/dense_97/MatMulMatMul)sequential_89/dense_96/Relu:activations:04sequential_89/dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_89/dense_97/MatMulÑ
-sequential_89/dense_97/BiasAdd/ReadVariableOpReadVariableOp6sequential_89_dense_97_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_89/dense_97/BiasAdd/ReadVariableOpÝ
sequential_89/dense_97/BiasAddBiasAdd'sequential_89/dense_97/MatMul:product:05sequential_89/dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2 
sequential_89/dense_97/BiasAdd
IdentityIdentity'sequential_89/dense_97/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identityµ
NoOpNoOp0^sequential_89/conv2d_144/BiasAdd/ReadVariableOp/^sequential_89/conv2d_144/Conv2D/ReadVariableOp0^sequential_89/conv2d_145/BiasAdd/ReadVariableOp/^sequential_89/conv2d_145/Conv2D/ReadVariableOp0^sequential_89/conv2d_146/BiasAdd/ReadVariableOp/^sequential_89/conv2d_146/Conv2D/ReadVariableOp.^sequential_89/dense_96/BiasAdd/ReadVariableOp-^sequential_89/dense_96/MatMul/ReadVariableOp.^sequential_89/dense_97/BiasAdd/ReadVariableOp-^sequential_89/dense_97/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : 2b
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
:ÿÿÿÿÿÿÿÿÿôô
-
_user_specified_namesequential_88_input
Õ
K
/__inference_random_flip_41_layer_call_fn_170479

inputs
identityë
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1683532
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í
¾
.__inference_sequential_88_layer_call_fn_168933
random_flip_41_input
unknown:	
	unknown_0:	
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_41_inputunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1689132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿôô: : : 22
StatefulPartitionedCallStatefulPartitionedCall:g c
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
.
_user_specified_namerandom_flip_41_input
ì
ÿ
F__inference_conv2d_146_layer_call_and_return_conditional_losses_169054

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ}} : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}} 
 
_user_specified_nameinputs
ü


.__inference_sequential_89_layer_call_fn_169852

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:
	unknown_6:	
	unknown_7:	

	unknown_8:

identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_89_layer_call_and_return_conditional_losses_1691102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
1:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
¨
j
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_168565

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
î
e
F__inference_dropout_37_layer_call_and_return_conditional_losses_170263

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *G­?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬>2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs
´
Ç
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_168818

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkipD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stateful_uniform/Const
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
stateful_uniform/Cast/x
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1Ù
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkip
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stack
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2Î
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_slice
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stack
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2Æ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1 
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg¸
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateful_uniform/StatelessRandomUniformV2
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub¯
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_1/y
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_2/y
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yª
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_5/y
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_6/y
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_7/y
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y°
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2Â
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_2
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack£
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1£
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2÷
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_2
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack£
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1£
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2÷
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Neg
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack£
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1£
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2ù
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_3
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack£
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1£
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2÷
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_3
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack£
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1£
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2÷
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack£
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1£
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2û
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1Ã
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
rotation_matrix/zeros/Constµ
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis¨
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
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
transform/fill_valueÉ
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿôô: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
¬

J__inference_random_flip_41_layer_call_and_return_conditional_losses_168448

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkip¢Cstateless_random_flip_left_right/assert_greater_equal/Assert/Assert¢Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert¢Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shape
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Const½
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prod
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/x¥
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkip¨
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stack¬
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1¬
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice´
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcast¬
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack°
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1°
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2ü
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1º
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg®
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

zeros_like
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
strided_slice/stack_2
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
&stateless_random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/Shape¿
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ26
4stateless_random_flip_left_right/strided_slice/stackº
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6stateless_random_flip_left_right/strided_slice/stack_1º
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2¤
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask20
.stateless_random_flip_left_right/strided_slice²
6stateless_random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 28
6stateless_random_flip_left_right/assert_positive/Const­
Astateless_random_flip_left_right/assert_positive/assert_less/LessLess?stateless_random_flip_left_right/assert_positive/Const:output:07stateless_random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:2C
Astateless_random_flip_left_right/assert_positive/assert_less/LessÒ
Bstateless_random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bstateless_random_flip_left_right/assert_positive/assert_less/Const·
@stateless_random_flip_left_right/assert_positive/assert_less/AllAllEstateless_random_flip_left_right/assert_positive/assert_less/Less:z:0Kstateless_random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2B
@stateless_random_flip_left_right/assert_positive/assert_less/All
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2K
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/Const
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2S
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0ë
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertIstateless_random_flip_left_right/assert_positive/assert_less/All:output:0Zstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2L
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert
%stateless_random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :2'
%stateless_random_flip_left_right/Rank´
7stateless_random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :29
7stateless_random_flip_left_right/assert_greater_equal/y«
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual.stateless_random_flip_left_right/Rank:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2D
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqualº
:stateless_random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2<
:stateless_random_flip_left_right/assert_greater_equal/RankÈ
Astateless_random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2C
Astateless_random_flip_left_right/assert_greater_equal/range/startÈ
Astateless_random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2C
Astateless_random_flip_left_right/assert_greater_equal/range/deltaú
;stateless_random_flip_left_right/assert_greater_equal/rangeRangeJstateless_random_flip_left_right/assert_greater_equal/range/start:output:0Cstateless_random_flip_left_right/assert_greater_equal/Rank:output:0Jstateless_random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2=
;stateless_random_flip_left_right/assert_greater_equal/range£
9stateless_random_flip_left_right/assert_greater_equal/AllAllFstateless_random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Dstateless_random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: 2;
9stateless_random_flip_left_right/assert_greater_equal/Allô
Bstateless_random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2D
Bstateless_random_flip_left_right/assert_greater_equal/Assert/Constø
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1û
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertAssertBstateless_random_flip_left_right/assert_greater_equal/All:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0.stateless_random_flip_left_right/Rank:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0K^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2E
Cstateless_random_flip_left_right/assert_greater_equal/Assert/Assert
3stateless_random_flip_left_right/control_dependencyIdentityinputsD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ25
3stateless_random_flip_left_right/control_dependencyÀ
(stateless_random_flip_left_right/Shape_1Shape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2*
(stateless_random_flip_left_right/Shape_1º
6stateless_random_flip_left_right/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6stateless_random_flip_left_right/strided_slice_1/stack¾
8stateless_random_flip_left_right/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8stateless_random_flip_left_right/strided_slice_1/stack_1¾
8stateless_random_flip_left_right/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8stateless_random_flip_left_right/strided_slice_1/stack_2´
0stateless_random_flip_left_right/strided_slice_1StridedSlice1stateless_random_flip_left_right/Shape_1:output:0?stateless_random_flip_left_right/strided_slice_1/stack:output:0Astateless_random_flip_left_right/strided_slice_1/stack_1:output:0Astateless_random_flip_left_right/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0stateless_random_flip_left_right/strided_slice_1ó
?stateless_random_flip_left_right/stateless_random_uniform/shapePack9stateless_random_flip_left_right/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shapeÃ
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/minÃ
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2?
=stateless_random_flip_left_right/stateless_random_uniform/maxÐ
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0D^stateless_random_flip_left_right/assert_greater_equal/Assert/Assert* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¬
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgÊ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2¶
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subÓ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=stateless_random_flip_left_right/stateless_random_uniform/mul¸
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9stateless_random_flip_left_right/stateless_random_uniform¦
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1¦
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2¦
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3
.stateless_random_flip_left_right/Reshape/shapePack9stateless_random_flip_left_right/strided_slice_1:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shape
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(stateless_random_flip_left_right/ReshapeÆ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_left_right/Round¬
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axis²
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2,
*stateless_random_flip_left_right/ReverseV2
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$stateless_random_flip_left_right/mul
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&stateless_random_flip_left_right/sub/xê
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stateless_random_flip_left_right/sub
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_left_right/mul_1
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$stateless_random_flip_left_right/add¦
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity·
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertCstateless_random_flip_left_right/assert_greater_equal/Assert/Assert2
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertJstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert2¢
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
e
I__inference_sequential_88_layer_call_and_return_conditional_losses_168574

inputs
identityð
random_flip_41/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1685592 
random_flip_41/PartitionedCall
"random_rotation_40/PartitionedCallPartitionedCall'random_flip_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1685652$
"random_rotation_40/PartitionedCall
random_zoom_40/PartitionedCallPartitionedCall+random_rotation_40/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1685712 
random_zoom_40/PartitionedCall
IdentityIdentity'random_zoom_40/PartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
¹ï
³
I__inference_sequential_88_layer_call_and_return_conditional_losses_170157

inputsN
@random_flip_41_stateful_uniform_full_int_rngreadandskip_resource:	I
;random_rotation_40_stateful_uniform_rngreadandskip_resource:	E
7random_zoom_40_stateful_uniform_rngreadandskip_resource:	
identity¢7random_flip_41/stateful_uniform_full_int/RngReadAndSkip¢^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢erandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢2random_rotation_40/stateful_uniform/RngReadAndSkip¢.random_zoom_40/stateful_uniform/RngReadAndSkipª
.random_flip_41/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_41/stateful_uniform_full_int/shapeª
.random_flip_41/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 20
.random_flip_41/stateful_uniform_full_int/Constù
-random_flip_41/stateful_uniform_full_int/ProdProd7random_flip_41/stateful_uniform_full_int/shape:output:07random_flip_41/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2/
-random_flip_41/stateful_uniform_full_int/Prod¤
/random_flip_41/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :21
/random_flip_41/stateful_uniform_full_int/Cast/xÒ
/random_flip_41/stateful_uniform_full_int/Cast_1Cast6random_flip_41/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/random_flip_41/stateful_uniform_full_int/Cast_1Ñ
7random_flip_41/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip@random_flip_41_stateful_uniform_full_int_rngreadandskip_resource8random_flip_41/stateful_uniform_full_int/Cast/x:output:03random_flip_41/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:29
7random_flip_41/stateful_uniform_full_int/RngReadAndSkipÆ
<random_flip_41/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<random_flip_41/stateful_uniform_full_int/strided_slice/stackÊ
>random_flip_41/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>random_flip_41/stateful_uniform_full_int/strided_slice/stack_1Ê
>random_flip_41/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>random_flip_41/stateful_uniform_full_int/strided_slice/stack_2Þ
6random_flip_41/stateful_uniform_full_int/strided_sliceStridedSlice?random_flip_41/stateful_uniform_full_int/RngReadAndSkip:value:0Erandom_flip_41/stateful_uniform_full_int/strided_slice/stack:output:0Grandom_flip_41/stateful_uniform_full_int/strided_slice/stack_1:output:0Grandom_flip_41/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask28
6random_flip_41/stateful_uniform_full_int/strided_sliceá
0random_flip_41/stateful_uniform_full_int/BitcastBitcast?random_flip_41/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type022
0random_flip_41/stateful_uniform_full_int/BitcastÊ
>random_flip_41/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2@
>random_flip_41/stateful_uniform_full_int/strided_slice_1/stackÎ
@random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1Î
@random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@random_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2Ö
8random_flip_41/stateful_uniform_full_int/strided_slice_1StridedSlice?random_flip_41/stateful_uniform_full_int/RngReadAndSkip:value:0Grandom_flip_41/stateful_uniform_full_int/strided_slice_1/stack:output:0Irandom_flip_41/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Irandom_flip_41/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2:
8random_flip_41/stateful_uniform_full_int/strided_slice_1ç
2random_flip_41/stateful_uniform_full_int/Bitcast_1BitcastArandom_flip_41/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type024
2random_flip_41/stateful_uniform_full_int/Bitcast_1
,random_flip_41/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2.
,random_flip_41/stateful_uniform_full_int/alg
(random_flip_41/stateful_uniform_full_intStatelessRandomUniformFullIntV27random_flip_41/stateful_uniform_full_int/shape:output:0;random_flip_41/stateful_uniform_full_int/Bitcast_1:output:09random_flip_41/stateful_uniform_full_int/Bitcast:output:05random_flip_41/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2*
(random_flip_41/stateful_uniform_full_int
random_flip_41/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip_41/zeros_like½
random_flip_41/stackPack1random_flip_41/stateful_uniform_full_int:output:0"random_flip_41/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_flip_41/stack
"random_flip_41/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"random_flip_41/strided_slice/stack
$random_flip_41/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$random_flip_41/strided_slice/stack_1
$random_flip_41/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$random_flip_41/strided_slice/stack_2â
random_flip_41/strided_sliceStridedSlicerandom_flip_41/stack:output:0+random_flip_41/strided_slice/stack:output:0-random_flip_41/strided_slice/stack_1:output:0-random_flip_41/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip_41/strided_sliceó
Brandom_flip_41/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2D
Brandom_flip_41/stateless_random_flip_left_right/control_dependencyé
5random_flip_41/stateless_random_flip_left_right/ShapeShapeKrandom_flip_41/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:27
5random_flip_41/stateless_random_flip_left_right/ShapeÔ
Crandom_flip_41/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Crandom_flip_41/stateless_random_flip_left_right/strided_slice/stackØ
Erandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Erandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_1Ø
Erandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Erandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_2
=random_flip_41/stateless_random_flip_left_right/strided_sliceStridedSlice>random_flip_41/stateless_random_flip_left_right/Shape:output:0Lrandom_flip_41/stateless_random_flip_left_right/strided_slice/stack:output:0Nrandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_1:output:0Nrandom_flip_41/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2?
=random_flip_41/stateless_random_flip_left_right/strided_slice
Nrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shapePackFrandom_flip_41/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2P
Nrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shapeá
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2N
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/miná
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2N
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/max·
erandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter%random_flip_41/strided_slice:output:0* 
_output_shapes
::2g
erandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterÙ
^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgf^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2`
^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¤
arandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Wrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0krandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0orandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0drandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2c
arandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2ò
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/subSubUrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Urandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2N
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/sub
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mulMuljrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Prandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
Lrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mulô
Hrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniformAddV2Prandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Urandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
Hrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniformÄ
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2A
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/1Ä
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2A
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/2Ä
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2A
?random_flip_41/stateless_random_flip_left_right/Reshape/shape/3Ú
=random_flip_41/stateless_random_flip_left_right/Reshape/shapePackFrandom_flip_41/stateless_random_flip_left_right/strided_slice:output:0Hrandom_flip_41/stateless_random_flip_left_right/Reshape/shape/1:output:0Hrandom_flip_41/stateless_random_flip_left_right/Reshape/shape/2:output:0Hrandom_flip_41/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2?
=random_flip_41/stateless_random_flip_left_right/Reshape/shapeÍ
7random_flip_41/stateless_random_flip_left_right/ReshapeReshapeLrandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform:z:0Frandom_flip_41/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7random_flip_41/stateless_random_flip_left_right/Reshapeó
5random_flip_41/stateless_random_flip_left_right/RoundRound@random_flip_41/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5random_flip_41/stateless_random_flip_left_right/RoundÊ
>random_flip_41/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2@
>random_flip_41/stateless_random_flip_left_right/ReverseV2/axisÕ
9random_flip_41/stateless_random_flip_left_right/ReverseV2	ReverseV2Krandom_flip_41/stateless_random_flip_left_right/control_dependency:output:0Grandom_flip_41/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2;
9random_flip_41/stateless_random_flip_left_right/ReverseV2¬
3random_flip_41/stateless_random_flip_left_right/mulMul9random_flip_41/stateless_random_flip_left_right/Round:y:0Brandom_flip_41/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô25
3random_flip_41/stateless_random_flip_left_right/mul³
5random_flip_41/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5random_flip_41/stateless_random_flip_left_right/sub/x¦
3random_flip_41/stateless_random_flip_left_right/subSub>random_flip_41/stateless_random_flip_left_right/sub/x:output:09random_flip_41/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3random_flip_41/stateless_random_flip_left_right/sub·
5random_flip_41/stateless_random_flip_left_right/mul_1Mul7random_flip_41/stateless_random_flip_left_right/sub:z:0Krandom_flip_41/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô27
5random_flip_41/stateless_random_flip_left_right/mul_1£
3random_flip_41/stateless_random_flip_left_right/addAddV27random_flip_41/stateless_random_flip_left_right/mul:z:09random_flip_41/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô25
3random_flip_41/stateless_random_flip_left_right/add
random_rotation_40/ShapeShape7random_flip_41/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
random_rotation_40/Shape
&random_rotation_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_rotation_40/strided_slice/stack
(random_rotation_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_40/strided_slice/stack_1
(random_rotation_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_40/strided_slice/stack_2Ô
 random_rotation_40/strided_sliceStridedSlice!random_rotation_40/Shape:output:0/random_rotation_40/strided_slice/stack:output:01random_rotation_40/strided_slice/stack_1:output:01random_rotation_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 random_rotation_40/strided_slice
(random_rotation_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_40/strided_slice_1/stack¢
*random_rotation_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_40/strided_slice_1/stack_1¢
*random_rotation_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_40/strided_slice_1/stack_2Þ
"random_rotation_40/strided_slice_1StridedSlice!random_rotation_40/Shape:output:01random_rotation_40/strided_slice_1/stack:output:03random_rotation_40/strided_slice_1/stack_1:output:03random_rotation_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_rotation_40/strided_slice_1
random_rotation_40/CastCast+random_rotation_40/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_40/Cast
(random_rotation_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(random_rotation_40/strided_slice_2/stack¢
*random_rotation_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_40/strided_slice_2/stack_1¢
*random_rotation_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*random_rotation_40/strided_slice_2/stack_2Þ
"random_rotation_40/strided_slice_2StridedSlice!random_rotation_40/Shape:output:01random_rotation_40/strided_slice_2/stack:output:03random_rotation_40/strided_slice_2/stack_1:output:03random_rotation_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"random_rotation_40/strided_slice_2
random_rotation_40/Cast_1Cast+random_rotation_40/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_40/Cast_1·
)random_rotation_40/stateful_uniform/shapePack)random_rotation_40/strided_slice:output:0*
N*
T0*
_output_shapes
:2+
)random_rotation_40/stateful_uniform/shape
'random_rotation_40/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_rotation_40/stateful_uniform/min
'random_rotation_40/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_rotation_40/stateful_uniform/max 
)random_rotation_40/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2+
)random_rotation_40/stateful_uniform/Constå
(random_rotation_40/stateful_uniform/ProdProd2random_rotation_40/stateful_uniform/shape:output:02random_rotation_40/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/stateful_uniform/Prod
*random_rotation_40/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2,
*random_rotation_40/stateful_uniform/Cast/xÃ
*random_rotation_40/stateful_uniform/Cast_1Cast1random_rotation_40/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2,
*random_rotation_40/stateful_uniform/Cast_1¸
2random_rotation_40/stateful_uniform/RngReadAndSkipRngReadAndSkip;random_rotation_40_stateful_uniform_rngreadandskip_resource3random_rotation_40/stateful_uniform/Cast/x:output:0.random_rotation_40/stateful_uniform/Cast_1:y:0*
_output_shapes
:24
2random_rotation_40/stateful_uniform/RngReadAndSkip¼
7random_rotation_40/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7random_rotation_40/stateful_uniform/strided_slice/stackÀ
9random_rotation_40/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9random_rotation_40/stateful_uniform/strided_slice/stack_1À
9random_rotation_40/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9random_rotation_40/stateful_uniform/strided_slice/stack_2À
1random_rotation_40/stateful_uniform/strided_sliceStridedSlice:random_rotation_40/stateful_uniform/RngReadAndSkip:value:0@random_rotation_40/stateful_uniform/strided_slice/stack:output:0Brandom_rotation_40/stateful_uniform/strided_slice/stack_1:output:0Brandom_rotation_40/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask23
1random_rotation_40/stateful_uniform/strided_sliceÒ
+random_rotation_40/stateful_uniform/BitcastBitcast:random_rotation_40/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02-
+random_rotation_40/stateful_uniform/BitcastÀ
9random_rotation_40/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2;
9random_rotation_40/stateful_uniform/strided_slice_1/stackÄ
;random_rotation_40/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_rotation_40/stateful_uniform/strided_slice_1/stack_1Ä
;random_rotation_40/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_rotation_40/stateful_uniform/strided_slice_1/stack_2¸
3random_rotation_40/stateful_uniform/strided_slice_1StridedSlice:random_rotation_40/stateful_uniform/RngReadAndSkip:value:0Brandom_rotation_40/stateful_uniform/strided_slice_1/stack:output:0Drandom_rotation_40/stateful_uniform/strided_slice_1/stack_1:output:0Drandom_rotation_40/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:25
3random_rotation_40/stateful_uniform/strided_slice_1Ø
-random_rotation_40/stateful_uniform/Bitcast_1Bitcast<random_rotation_40/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02/
-random_rotation_40/stateful_uniform/Bitcast_1Æ
@random_rotation_40/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2B
@random_rotation_40/stateful_uniform/StatelessRandomUniformV2/algª
<random_rotation_40/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV22random_rotation_40/stateful_uniform/shape:output:06random_rotation_40/stateful_uniform/Bitcast_1:output:04random_rotation_40/stateful_uniform/Bitcast:output:0Irandom_rotation_40/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<random_rotation_40/stateful_uniform/StatelessRandomUniformV2Þ
'random_rotation_40/stateful_uniform/subSub0random_rotation_40/stateful_uniform/max:output:00random_rotation_40/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2)
'random_rotation_40/stateful_uniform/subû
'random_rotation_40/stateful_uniform/mulMulErandom_rotation_40/stateful_uniform/StatelessRandomUniformV2:output:0+random_rotation_40/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation_40/stateful_uniform/mulà
#random_rotation_40/stateful_uniformAddV2+random_rotation_40/stateful_uniform/mul:z:00random_rotation_40/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation_40/stateful_uniform
(random_rotation_40/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(random_rotation_40/rotation_matrix/sub/yÊ
&random_rotation_40/rotation_matrix/subSubrandom_rotation_40/Cast_1:y:01random_rotation_40/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2(
&random_rotation_40/rotation_matrix/sub®
&random_rotation_40/rotation_matrix/CosCos'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation_40/rotation_matrix/Cos
*random_rotation_40/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_40/rotation_matrix/sub_1/yÐ
(random_rotation_40/rotation_matrix/sub_1Subrandom_rotation_40/Cast_1:y:03random_rotation_40/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/rotation_matrix/sub_1ß
&random_rotation_40/rotation_matrix/mulMul*random_rotation_40/rotation_matrix/Cos:y:0,random_rotation_40/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation_40/rotation_matrix/mul®
&random_rotation_40/rotation_matrix/SinSin'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation_40/rotation_matrix/Sin
*random_rotation_40/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_40/rotation_matrix/sub_2/yÎ
(random_rotation_40/rotation_matrix/sub_2Subrandom_rotation_40/Cast:y:03random_rotation_40/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/rotation_matrix/sub_2ã
(random_rotation_40/rotation_matrix/mul_1Mul*random_rotation_40/rotation_matrix/Sin:y:0,random_rotation_40/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/mul_1ã
(random_rotation_40/rotation_matrix/sub_3Sub*random_rotation_40/rotation_matrix/mul:z:0,random_rotation_40/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/sub_3ã
(random_rotation_40/rotation_matrix/sub_4Sub*random_rotation_40/rotation_matrix/sub:z:0,random_rotation_40/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/sub_4¡
,random_rotation_40/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,random_rotation_40/rotation_matrix/truediv/yö
*random_rotation_40/rotation_matrix/truedivRealDiv,random_rotation_40/rotation_matrix/sub_4:z:05random_rotation_40/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*random_rotation_40/rotation_matrix/truediv
*random_rotation_40/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_40/rotation_matrix/sub_5/yÎ
(random_rotation_40/rotation_matrix/sub_5Subrandom_rotation_40/Cast:y:03random_rotation_40/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/rotation_matrix/sub_5²
(random_rotation_40/rotation_matrix/Sin_1Sin'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/Sin_1
*random_rotation_40/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_40/rotation_matrix/sub_6/yÐ
(random_rotation_40/rotation_matrix/sub_6Subrandom_rotation_40/Cast_1:y:03random_rotation_40/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/rotation_matrix/sub_6å
(random_rotation_40/rotation_matrix/mul_2Mul,random_rotation_40/rotation_matrix/Sin_1:y:0,random_rotation_40/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/mul_2²
(random_rotation_40/rotation_matrix/Cos_1Cos'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/Cos_1
*random_rotation_40/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*random_rotation_40/rotation_matrix/sub_7/yÎ
(random_rotation_40/rotation_matrix/sub_7Subrandom_rotation_40/Cast:y:03random_rotation_40/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2*
(random_rotation_40/rotation_matrix/sub_7å
(random_rotation_40/rotation_matrix/mul_3Mul,random_rotation_40/rotation_matrix/Cos_1:y:0,random_rotation_40/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/mul_3ã
&random_rotation_40/rotation_matrix/addAddV2,random_rotation_40/rotation_matrix/mul_2:z:0,random_rotation_40/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation_40/rotation_matrix/addã
(random_rotation_40/rotation_matrix/sub_8Sub,random_rotation_40/rotation_matrix/sub_5:z:0*random_rotation_40/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/sub_8¥
.random_rotation_40/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @20
.random_rotation_40/rotation_matrix/truediv_1/yü
,random_rotation_40/rotation_matrix/truediv_1RealDiv,random_rotation_40/rotation_matrix/sub_8:z:07random_rotation_40/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,random_rotation_40/rotation_matrix/truediv_1«
(random_rotation_40/rotation_matrix/ShapeShape'random_rotation_40/stateful_uniform:z:0*
T0*
_output_shapes
:2*
(random_rotation_40/rotation_matrix/Shapeº
6random_rotation_40/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_rotation_40/rotation_matrix/strided_slice/stack¾
8random_rotation_40/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_40/rotation_matrix/strided_slice/stack_1¾
8random_rotation_40/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_40/rotation_matrix/strided_slice/stack_2´
0random_rotation_40/rotation_matrix/strided_sliceStridedSlice1random_rotation_40/rotation_matrix/Shape:output:0?random_rotation_40/rotation_matrix/strided_slice/stack:output:0Arandom_rotation_40/rotation_matrix/strided_slice/stack_1:output:0Arandom_rotation_40/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_rotation_40/rotation_matrix/strided_slice²
(random_rotation_40/rotation_matrix/Cos_2Cos'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/Cos_2Å
8random_rotation_40/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_1/stackÉ
:random_rotation_40/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_1/stack_1É
:random_rotation_40/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_1/stack_2é
2random_rotation_40/rotation_matrix/strided_slice_1StridedSlice,random_rotation_40/rotation_matrix/Cos_2:y:0Arandom_rotation_40/rotation_matrix/strided_slice_1/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_1/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_1²
(random_rotation_40/rotation_matrix/Sin_2Sin'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/Sin_2Å
8random_rotation_40/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_2/stackÉ
:random_rotation_40/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_2/stack_1É
:random_rotation_40/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_2/stack_2é
2random_rotation_40/rotation_matrix/strided_slice_2StridedSlice,random_rotation_40/rotation_matrix/Sin_2:y:0Arandom_rotation_40/rotation_matrix/strided_slice_2/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_2/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_2Æ
&random_rotation_40/rotation_matrix/NegNeg;random_rotation_40/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation_40/rotation_matrix/NegÅ
8random_rotation_40/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_3/stackÉ
:random_rotation_40/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_3/stack_1É
:random_rotation_40/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_3/stack_2ë
2random_rotation_40/rotation_matrix/strided_slice_3StridedSlice.random_rotation_40/rotation_matrix/truediv:z:0Arandom_rotation_40/rotation_matrix/strided_slice_3/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_3/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_3²
(random_rotation_40/rotation_matrix/Sin_3Sin'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/Sin_3Å
8random_rotation_40/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_4/stackÉ
:random_rotation_40/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_4/stack_1É
:random_rotation_40/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_4/stack_2é
2random_rotation_40/rotation_matrix/strided_slice_4StridedSlice,random_rotation_40/rotation_matrix/Sin_3:y:0Arandom_rotation_40/rotation_matrix/strided_slice_4/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_4/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_4²
(random_rotation_40/rotation_matrix/Cos_3Cos'random_rotation_40/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/Cos_3Å
8random_rotation_40/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_5/stackÉ
:random_rotation_40/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_5/stack_1É
:random_rotation_40/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_5/stack_2é
2random_rotation_40/rotation_matrix/strided_slice_5StridedSlice,random_rotation_40/rotation_matrix/Cos_3:y:0Arandom_rotation_40/rotation_matrix/strided_slice_5/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_5/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_5Å
8random_rotation_40/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2:
8random_rotation_40/rotation_matrix/strided_slice_6/stackÉ
:random_rotation_40/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2<
:random_rotation_40/rotation_matrix/strided_slice_6/stack_1É
:random_rotation_40/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2<
:random_rotation_40/rotation_matrix/strided_slice_6/stack_2í
2random_rotation_40/rotation_matrix/strided_slice_6StridedSlice0random_rotation_40/rotation_matrix/truediv_1:z:0Arandom_rotation_40/rotation_matrix/strided_slice_6/stack:output:0Crandom_rotation_40/rotation_matrix/strided_slice_6/stack_1:output:0Crandom_rotation_40/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask24
2random_rotation_40/rotation_matrix/strided_slice_6¨
1random_rotation_40/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_rotation_40/rotation_matrix/zeros/packed/1
/random_rotation_40/rotation_matrix/zeros/packedPack9random_rotation_40/rotation_matrix/strided_slice:output:0:random_rotation_40/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:21
/random_rotation_40/rotation_matrix/zeros/packed¥
.random_rotation_40/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.random_rotation_40/rotation_matrix/zeros/Const
(random_rotation_40/rotation_matrix/zerosFill8random_rotation_40/rotation_matrix/zeros/packed:output:07random_rotation_40/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_rotation_40/rotation_matrix/zeros¢
.random_rotation_40/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation_40/rotation_matrix/concat/axisæ
)random_rotation_40/rotation_matrix/concatConcatV2;random_rotation_40/rotation_matrix/strided_slice_1:output:0*random_rotation_40/rotation_matrix/Neg:y:0;random_rotation_40/rotation_matrix/strided_slice_3:output:0;random_rotation_40/rotation_matrix/strided_slice_4:output:0;random_rotation_40/rotation_matrix/strided_slice_5:output:0;random_rotation_40/rotation_matrix/strided_slice_6:output:01random_rotation_40/rotation_matrix/zeros:output:07random_rotation_40/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)random_rotation_40/rotation_matrix/concat¯
"random_rotation_40/transform/ShapeShape7random_flip_41/stateless_random_flip_left_right/add:z:0*
T0*
_output_shapes
:2$
"random_rotation_40/transform/Shape®
0random_rotation_40/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0random_rotation_40/transform/strided_slice/stack²
2random_rotation_40/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2random_rotation_40/transform/strided_slice/stack_1²
2random_rotation_40/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2random_rotation_40/transform/strided_slice/stack_2ü
*random_rotation_40/transform/strided_sliceStridedSlice+random_rotation_40/transform/Shape:output:09random_rotation_40/transform/strided_slice/stack:output:0;random_rotation_40/transform/strided_slice/stack_1:output:0;random_rotation_40/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*random_rotation_40/transform/strided_slice
'random_rotation_40/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'random_rotation_40/transform/fill_valueÙ
7random_rotation_40/transform/ImageProjectiveTransformV3ImageProjectiveTransformV37random_flip_41/stateless_random_flip_left_right/add:z:02random_rotation_40/rotation_matrix/concat:output:03random_rotation_40/transform/strided_slice:output:00random_rotation_40/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR29
7random_rotation_40/transform/ImageProjectiveTransformV3¨
random_zoom_40/ShapeShapeLrandom_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2
random_zoom_40/Shape
"random_zoom_40/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"random_zoom_40/strided_slice/stack
$random_zoom_40/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$random_zoom_40/strided_slice/stack_1
$random_zoom_40/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$random_zoom_40/strided_slice/stack_2¼
random_zoom_40/strided_sliceStridedSlicerandom_zoom_40/Shape:output:0+random_zoom_40/strided_slice/stack:output:0-random_zoom_40/strided_slice/stack_1:output:0-random_zoom_40/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_zoom_40/strided_slice
$random_zoom_40/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$random_zoom_40/strided_slice_1/stack
&random_zoom_40/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&random_zoom_40/strided_slice_1/stack_1
&random_zoom_40/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&random_zoom_40/strided_slice_1/stack_2Æ
random_zoom_40/strided_slice_1StridedSlicerandom_zoom_40/Shape:output:0-random_zoom_40/strided_slice_1/stack:output:0/random_zoom_40/strided_slice_1/stack_1:output:0/random_zoom_40/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
random_zoom_40/strided_slice_1
random_zoom_40/CastCast'random_zoom_40/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_40/Cast
$random_zoom_40/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$random_zoom_40/strided_slice_2/stack
&random_zoom_40/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&random_zoom_40/strided_slice_2/stack_1
&random_zoom_40/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&random_zoom_40/strided_slice_2/stack_2Æ
random_zoom_40/strided_slice_2StridedSlicerandom_zoom_40/Shape:output:0-random_zoom_40/strided_slice_2/stack:output:0/random_zoom_40/strided_slice_2/stack_1:output:0/random_zoom_40/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
random_zoom_40/strided_slice_2
random_zoom_40/Cast_1Cast'random_zoom_40/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_zoom_40/Cast_1
'random_zoom_40/stateful_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'random_zoom_40/stateful_uniform/shape/1Ý
%random_zoom_40/stateful_uniform/shapePack%random_zoom_40/strided_slice:output:00random_zoom_40/stateful_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%random_zoom_40/stateful_uniform/shape
#random_zoom_40/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2%
#random_zoom_40/stateful_uniform/min
#random_zoom_40/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?2%
#random_zoom_40/stateful_uniform/max
%random_zoom_40/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2'
%random_zoom_40/stateful_uniform/ConstÕ
$random_zoom_40/stateful_uniform/ProdProd.random_zoom_40/stateful_uniform/shape:output:0.random_zoom_40/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2&
$random_zoom_40/stateful_uniform/Prod
&random_zoom_40/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom_40/stateful_uniform/Cast/x·
&random_zoom_40/stateful_uniform/Cast_1Cast-random_zoom_40/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2(
&random_zoom_40/stateful_uniform/Cast_1¤
.random_zoom_40/stateful_uniform/RngReadAndSkipRngReadAndSkip7random_zoom_40_stateful_uniform_rngreadandskip_resource/random_zoom_40/stateful_uniform/Cast/x:output:0*random_zoom_40/stateful_uniform/Cast_1:y:0*
_output_shapes
:20
.random_zoom_40/stateful_uniform/RngReadAndSkip´
3random_zoom_40/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_zoom_40/stateful_uniform/strided_slice/stack¸
5random_zoom_40/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_zoom_40/stateful_uniform/strided_slice/stack_1¸
5random_zoom_40/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_zoom_40/stateful_uniform/strided_slice/stack_2¨
-random_zoom_40/stateful_uniform/strided_sliceStridedSlice6random_zoom_40/stateful_uniform/RngReadAndSkip:value:0<random_zoom_40/stateful_uniform/strided_slice/stack:output:0>random_zoom_40/stateful_uniform/strided_slice/stack_1:output:0>random_zoom_40/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2/
-random_zoom_40/stateful_uniform/strided_sliceÆ
'random_zoom_40/stateful_uniform/BitcastBitcast6random_zoom_40/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02)
'random_zoom_40/stateful_uniform/Bitcast¸
5random_zoom_40/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5random_zoom_40/stateful_uniform/strided_slice_1/stack¼
7random_zoom_40/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_zoom_40/stateful_uniform/strided_slice_1/stack_1¼
7random_zoom_40/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_zoom_40/stateful_uniform/strided_slice_1/stack_2 
/random_zoom_40/stateful_uniform/strided_slice_1StridedSlice6random_zoom_40/stateful_uniform/RngReadAndSkip:value:0>random_zoom_40/stateful_uniform/strided_slice_1/stack:output:0@random_zoom_40/stateful_uniform/strided_slice_1/stack_1:output:0@random_zoom_40/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:21
/random_zoom_40/stateful_uniform/strided_slice_1Ì
)random_zoom_40/stateful_uniform/Bitcast_1Bitcast8random_zoom_40/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02+
)random_zoom_40/stateful_uniform/Bitcast_1¾
<random_zoom_40/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2>
<random_zoom_40/stateful_uniform/StatelessRandomUniformV2/alg
8random_zoom_40/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2.random_zoom_40/stateful_uniform/shape:output:02random_zoom_40/stateful_uniform/Bitcast_1:output:00random_zoom_40/stateful_uniform/Bitcast:output:0Erandom_zoom_40/stateful_uniform/StatelessRandomUniformV2/alg:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8random_zoom_40/stateful_uniform/StatelessRandomUniformV2Î
#random_zoom_40/stateful_uniform/subSub,random_zoom_40/stateful_uniform/max:output:0,random_zoom_40/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2%
#random_zoom_40/stateful_uniform/subï
#random_zoom_40/stateful_uniform/mulMulArandom_zoom_40/stateful_uniform/StatelessRandomUniformV2:output:0'random_zoom_40/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_zoom_40/stateful_uniform/mulÔ
random_zoom_40/stateful_uniformAddV2'random_zoom_40/stateful_uniform/mul:z:0,random_zoom_40/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
random_zoom_40/stateful_uniformz
random_zoom_40/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
random_zoom_40/concat/axisä
random_zoom_40/concatConcatV2#random_zoom_40/stateful_uniform:z:0#random_zoom_40/stateful_uniform:z:0#random_zoom_40/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_zoom_40/concat
 random_zoom_40/zoom_matrix/ShapeShaperandom_zoom_40/concat:output:0*
T0*
_output_shapes
:2"
 random_zoom_40/zoom_matrix/Shapeª
.random_zoom_40/zoom_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.random_zoom_40/zoom_matrix/strided_slice/stack®
0random_zoom_40/zoom_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0random_zoom_40/zoom_matrix/strided_slice/stack_1®
0random_zoom_40/zoom_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0random_zoom_40/zoom_matrix/strided_slice/stack_2
(random_zoom_40/zoom_matrix/strided_sliceStridedSlice)random_zoom_40/zoom_matrix/Shape:output:07random_zoom_40/zoom_matrix/strided_slice/stack:output:09random_zoom_40/zoom_matrix/strided_slice/stack_1:output:09random_zoom_40/zoom_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(random_zoom_40/zoom_matrix/strided_slice
 random_zoom_40/zoom_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2"
 random_zoom_40/zoom_matrix/sub/y®
random_zoom_40/zoom_matrix/subSubrandom_zoom_40/Cast_1:y:0)random_zoom_40/zoom_matrix/sub/y:output:0*
T0*
_output_shapes
: 2 
random_zoom_40/zoom_matrix/sub
$random_zoom_40/zoom_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$random_zoom_40/zoom_matrix/truediv/yÇ
"random_zoom_40/zoom_matrix/truedivRealDiv"random_zoom_40/zoom_matrix/sub:z:0-random_zoom_40/zoom_matrix/truediv/y:output:0*
T0*
_output_shapes
: 2$
"random_zoom_40/zoom_matrix/truediv¹
0random_zoom_40/zoom_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*!
valueB"            22
0random_zoom_40/zoom_matrix/strided_slice_1/stack½
2random_zoom_40/zoom_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           24
2random_zoom_40/zoom_matrix/strided_slice_1/stack_1½
2random_zoom_40/zoom_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         24
2random_zoom_40/zoom_matrix/strided_slice_1/stack_2Ë
*random_zoom_40/zoom_matrix/strided_slice_1StridedSlicerandom_zoom_40/concat:output:09random_zoom_40/zoom_matrix/strided_slice_1/stack:output:0;random_zoom_40/zoom_matrix/strided_slice_1/stack_1:output:0;random_zoom_40/zoom_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2,
*random_zoom_40/zoom_matrix/strided_slice_1
"random_zoom_40/zoom_matrix/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"random_zoom_40/zoom_matrix/sub_1/xß
 random_zoom_40/zoom_matrix/sub_1Sub+random_zoom_40/zoom_matrix/sub_1/x:output:03random_zoom_40/zoom_matrix/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_zoom_40/zoom_matrix/sub_1Ç
random_zoom_40/zoom_matrix/mulMul&random_zoom_40/zoom_matrix/truediv:z:0$random_zoom_40/zoom_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
random_zoom_40/zoom_matrix/mul
"random_zoom_40/zoom_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"random_zoom_40/zoom_matrix/sub_2/y²
 random_zoom_40/zoom_matrix/sub_2Subrandom_zoom_40/Cast:y:0+random_zoom_40/zoom_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2"
 random_zoom_40/zoom_matrix/sub_2
&random_zoom_40/zoom_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&random_zoom_40/zoom_matrix/truediv_1/yÏ
$random_zoom_40/zoom_matrix/truediv_1RealDiv$random_zoom_40/zoom_matrix/sub_2:z:0/random_zoom_40/zoom_matrix/truediv_1/y:output:0*
T0*
_output_shapes
: 2&
$random_zoom_40/zoom_matrix/truediv_1¹
0random_zoom_40/zoom_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*!
valueB"           22
0random_zoom_40/zoom_matrix/strided_slice_2/stack½
2random_zoom_40/zoom_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           24
2random_zoom_40/zoom_matrix/strided_slice_2/stack_1½
2random_zoom_40/zoom_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         24
2random_zoom_40/zoom_matrix/strided_slice_2/stack_2Ë
*random_zoom_40/zoom_matrix/strided_slice_2StridedSlicerandom_zoom_40/concat:output:09random_zoom_40/zoom_matrix/strided_slice_2/stack:output:0;random_zoom_40/zoom_matrix/strided_slice_2/stack_1:output:0;random_zoom_40/zoom_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2,
*random_zoom_40/zoom_matrix/strided_slice_2
"random_zoom_40/zoom_matrix/sub_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2$
"random_zoom_40/zoom_matrix/sub_3/xß
 random_zoom_40/zoom_matrix/sub_3Sub+random_zoom_40/zoom_matrix/sub_3/x:output:03random_zoom_40/zoom_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_zoom_40/zoom_matrix/sub_3Í
 random_zoom_40/zoom_matrix/mul_1Mul(random_zoom_40/zoom_matrix/truediv_1:z:0$random_zoom_40/zoom_matrix/sub_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_zoom_40/zoom_matrix/mul_1¹
0random_zoom_40/zoom_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*!
valueB"            22
0random_zoom_40/zoom_matrix/strided_slice_3/stack½
2random_zoom_40/zoom_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           24
2random_zoom_40/zoom_matrix/strided_slice_3/stack_1½
2random_zoom_40/zoom_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         24
2random_zoom_40/zoom_matrix/strided_slice_3/stack_2Ë
*random_zoom_40/zoom_matrix/strided_slice_3StridedSlicerandom_zoom_40/concat:output:09random_zoom_40/zoom_matrix/strided_slice_3/stack:output:0;random_zoom_40/zoom_matrix/strided_slice_3/stack_1:output:0;random_zoom_40/zoom_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2,
*random_zoom_40/zoom_matrix/strided_slice_3
)random_zoom_40/zoom_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)random_zoom_40/zoom_matrix/zeros/packed/1ï
'random_zoom_40/zoom_matrix/zeros/packedPack1random_zoom_40/zoom_matrix/strided_slice:output:02random_zoom_40/zoom_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'random_zoom_40/zoom_matrix/zeros/packed
&random_zoom_40/zoom_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&random_zoom_40/zoom_matrix/zeros/Constá
 random_zoom_40/zoom_matrix/zerosFill0random_zoom_40/zoom_matrix/zeros/packed:output:0/random_zoom_40/zoom_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_zoom_40/zoom_matrix/zeros
+random_zoom_40/zoom_matrix/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+random_zoom_40/zoom_matrix/zeros_1/packed/1õ
)random_zoom_40/zoom_matrix/zeros_1/packedPack1random_zoom_40/zoom_matrix/strided_slice:output:04random_zoom_40/zoom_matrix/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2+
)random_zoom_40/zoom_matrix/zeros_1/packed
(random_zoom_40/zoom_matrix/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(random_zoom_40/zoom_matrix/zeros_1/Consté
"random_zoom_40/zoom_matrix/zeros_1Fill2random_zoom_40/zoom_matrix/zeros_1/packed:output:01random_zoom_40/zoom_matrix/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"random_zoom_40/zoom_matrix/zeros_1¹
0random_zoom_40/zoom_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*!
valueB"           22
0random_zoom_40/zoom_matrix/strided_slice_4/stack½
2random_zoom_40/zoom_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           24
2random_zoom_40/zoom_matrix/strided_slice_4/stack_1½
2random_zoom_40/zoom_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         24
2random_zoom_40/zoom_matrix/strided_slice_4/stack_2Ë
*random_zoom_40/zoom_matrix/strided_slice_4StridedSlicerandom_zoom_40/concat:output:09random_zoom_40/zoom_matrix/strided_slice_4/stack:output:0;random_zoom_40/zoom_matrix/strided_slice_4/stack_1:output:0;random_zoom_40/zoom_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask*
shrink_axis_mask2,
*random_zoom_40/zoom_matrix/strided_slice_4
+random_zoom_40/zoom_matrix/zeros_2/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+random_zoom_40/zoom_matrix/zeros_2/packed/1õ
)random_zoom_40/zoom_matrix/zeros_2/packedPack1random_zoom_40/zoom_matrix/strided_slice:output:04random_zoom_40/zoom_matrix/zeros_2/packed/1:output:0*
N*
T0*
_output_shapes
:2+
)random_zoom_40/zoom_matrix/zeros_2/packed
(random_zoom_40/zoom_matrix/zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(random_zoom_40/zoom_matrix/zeros_2/Consté
"random_zoom_40/zoom_matrix/zeros_2Fill2random_zoom_40/zoom_matrix/zeros_2/packed:output:01random_zoom_40/zoom_matrix/zeros_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"random_zoom_40/zoom_matrix/zeros_2
&random_zoom_40/zoom_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&random_zoom_40/zoom_matrix/concat/axis÷
!random_zoom_40/zoom_matrix/concatConcatV23random_zoom_40/zoom_matrix/strided_slice_3:output:0)random_zoom_40/zoom_matrix/zeros:output:0"random_zoom_40/zoom_matrix/mul:z:0+random_zoom_40/zoom_matrix/zeros_1:output:03random_zoom_40/zoom_matrix/strided_slice_4:output:0$random_zoom_40/zoom_matrix/mul_1:z:0+random_zoom_40/zoom_matrix/zeros_2:output:0/random_zoom_40/zoom_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!random_zoom_40/zoom_matrix/concat¼
random_zoom_40/transform/ShapeShapeLrandom_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*
_output_shapes
:2 
random_zoom_40/transform/Shape¦
,random_zoom_40/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,random_zoom_40/transform/strided_slice/stackª
.random_zoom_40/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.random_zoom_40/transform/strided_slice/stack_1ª
.random_zoom_40/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.random_zoom_40/transform/strided_slice/stack_2ä
&random_zoom_40/transform/strided_sliceStridedSlice'random_zoom_40/transform/Shape:output:05random_zoom_40/transform/strided_slice/stack:output:07random_zoom_40/transform/strided_slice/stack_1:output:07random_zoom_40/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2(
&random_zoom_40/transform/strided_slice
#random_zoom_40/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#random_zoom_40/transform/fill_valueÖ
3random_zoom_40/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3Lrandom_rotation_40/transform/ImageProjectiveTransformV3:transformed_images:0*random_zoom_40/zoom_matrix/concat:output:0/random_zoom_40/transform/strided_slice:output:0,random_zoom_40/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR25
3random_zoom_40/transform/ImageProjectiveTransformV3­
IdentityIdentityHrandom_zoom_40/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity·
NoOpNoOp8^random_flip_41/stateful_uniform_full_int/RngReadAndSkip_^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgf^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter3^random_rotation_40/stateful_uniform/RngReadAndSkip/^random_zoom_40/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿôô: : : 2r
7random_flip_41/stateful_uniform_full_int/RngReadAndSkip7random_flip_41/stateful_uniform_full_int/RngReadAndSkip2À
^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg^random_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2Î
erandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCountererandom_flip_41/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2h
2random_rotation_40/stateful_uniform/RngReadAndSkip2random_rotation_40/stateful_uniform/RngReadAndSkip2`
.random_zoom_40/stateful_uniform/RngReadAndSkip.random_zoom_40/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ø
ÿ
F__inference_conv2d_145_layer_call_and_return_conditional_losses_170217

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿúú: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
ø
O
3__inference_random_rotation_40_layer_call_fn_170625

inputs
identityÖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1685652
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs

d
F__inference_dropout_37_layer_call_and_return_conditional_losses_169066

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>>@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs
£

.__inference_sequential_89_layer_call_fn_169133
sequential_88_input!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@
	unknown_5:
	unknown_6:	
	unknown_7:	

	unknown_8:

identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallsequential_88_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_89_layer_call_and_return_conditional_losses_1691102
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
1:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
-
_user_specified_namesequential_88_input
¬

J__inference_random_flip_41_layer_call_and_return_conditional_losses_170412

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkip¢Cstateless_random_flip_left_right/assert_greater_equal/Assert/Assert¢Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert¢Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shape
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Const½
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prod
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/x¥
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkip¨
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stack¬
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1¬
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice´
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcast¬
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack°
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1°
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2ü
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1º
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg®
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

zeros_like
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
strided_slice/stack_2
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
&stateless_random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/Shape¿
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ26
4stateless_random_flip_left_right/strided_slice/stackº
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6stateless_random_flip_left_right/strided_slice/stack_1º
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2¤
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask20
.stateless_random_flip_left_right/strided_slice²
6stateless_random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 28
6stateless_random_flip_left_right/assert_positive/Const­
Astateless_random_flip_left_right/assert_positive/assert_less/LessLess?stateless_random_flip_left_right/assert_positive/Const:output:07stateless_random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:2C
Astateless_random_flip_left_right/assert_positive/assert_less/LessÒ
Bstateless_random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bstateless_random_flip_left_right/assert_positive/assert_less/Const·
@stateless_random_flip_left_right/assert_positive/assert_less/AllAllEstateless_random_flip_left_right/assert_positive/assert_less/Less:z:0Kstateless_random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2B
@stateless_random_flip_left_right/assert_positive/assert_less/All
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2K
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/Const
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2S
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0ë
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertIstateless_random_flip_left_right/assert_positive/assert_less/All:output:0Zstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2L
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert
%stateless_random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :2'
%stateless_random_flip_left_right/Rank´
7stateless_random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :29
7stateless_random_flip_left_right/assert_greater_equal/y«
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual.stateless_random_flip_left_right/Rank:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2D
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqualº
:stateless_random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2<
:stateless_random_flip_left_right/assert_greater_equal/RankÈ
Astateless_random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2C
Astateless_random_flip_left_right/assert_greater_equal/range/startÈ
Astateless_random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2C
Astateless_random_flip_left_right/assert_greater_equal/range/deltaú
;stateless_random_flip_left_right/assert_greater_equal/rangeRangeJstateless_random_flip_left_right/assert_greater_equal/range/start:output:0Cstateless_random_flip_left_right/assert_greater_equal/Rank:output:0Jstateless_random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2=
;stateless_random_flip_left_right/assert_greater_equal/range£
9stateless_random_flip_left_right/assert_greater_equal/AllAllFstateless_random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Dstateless_random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: 2;
9stateless_random_flip_left_right/assert_greater_equal/Allô
Bstateless_random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2D
Bstateless_random_flip_left_right/assert_greater_equal/Assert/Constø
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1û
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertAssertBstateless_random_flip_left_right/assert_greater_equal/All:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0.stateless_random_flip_left_right/Rank:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0K^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2E
Cstateless_random_flip_left_right/assert_greater_equal/Assert/Assert
3stateless_random_flip_left_right/control_dependencyIdentityinputsD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ25
3stateless_random_flip_left_right/control_dependencyÀ
(stateless_random_flip_left_right/Shape_1Shape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2*
(stateless_random_flip_left_right/Shape_1º
6stateless_random_flip_left_right/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6stateless_random_flip_left_right/strided_slice_1/stack¾
8stateless_random_flip_left_right/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8stateless_random_flip_left_right/strided_slice_1/stack_1¾
8stateless_random_flip_left_right/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8stateless_random_flip_left_right/strided_slice_1/stack_2´
0stateless_random_flip_left_right/strided_slice_1StridedSlice1stateless_random_flip_left_right/Shape_1:output:0?stateless_random_flip_left_right/strided_slice_1/stack:output:0Astateless_random_flip_left_right/strided_slice_1/stack_1:output:0Astateless_random_flip_left_right/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0stateless_random_flip_left_right/strided_slice_1ó
?stateless_random_flip_left_right/stateless_random_uniform/shapePack9stateless_random_flip_left_right/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shapeÃ
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/minÃ
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2?
=stateless_random_flip_left_right/stateless_random_uniform/maxÐ
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0D^stateless_random_flip_left_right/assert_greater_equal/Assert/Assert* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¬
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgÊ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2¶
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subÓ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=stateless_random_flip_left_right/stateless_random_uniform/mul¸
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9stateless_random_flip_left_right/stateless_random_uniform¦
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1¦
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2¦
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3
.stateless_random_flip_left_right/Reshape/shapePack9stateless_random_flip_left_right/strided_slice_1:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shape
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(stateless_random_flip_left_right/ReshapeÆ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_left_right/Round¬
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axis²
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2,
*stateless_random_flip_left_right/ReverseV2
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$stateless_random_flip_left_right/mul
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&stateless_random_flip_left_right/sub/xê
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stateless_random_flip_left_right/sub
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_left_right/mul_1
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$stateless_random_flip_left_right/add¦
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity·
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertCstateless_random_flip_left_right/assert_greater_equal/Assert/Assert2
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertJstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert2¢
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü8
Ö
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
dense_96_169414:
dense_96_169416:	"
dense_97_169419:	

dense_97_169421:

identity¢"conv2d_144/StatefulPartitionedCall¢"conv2d_145/StatefulPartitionedCall¢"conv2d_146/StatefulPartitionedCall¢ dense_96/StatefulPartitionedCall¢ dense_97/StatefulPartitionedCall¢"dropout_37/StatefulPartitionedCall¢%sequential_88/StatefulPartitionedCall×
%sequential_88/StatefulPartitionedCallStatefulPartitionedCallsequential_88_inputsequential_88_169386sequential_88_169388sequential_88_169390*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1689132'
%sequential_88/StatefulPartitionedCall
rescaling_51/PartitionedCallPartitionedCall.sequential_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_1690052
rescaling_51/PartitionedCallÇ
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall%rescaling_51/PartitionedCall:output:0conv2d_144_169394conv2d_144_169396*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_144_layer_call_and_return_conditional_losses_1690182$
"conv2d_144/StatefulPartitionedCall
!max_pooling2d_144/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_1689592#
!max_pooling2d_144/PartitionedCallÌ
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_144/PartitionedCall:output:0conv2d_145_169400conv2d_145_169402*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_145_layer_call_and_return_conditional_losses_1690362$
"conv2d_145/StatefulPartitionedCall
!max_pooling2d_145/PartitionedCallPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}} * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_1689712#
!max_pooling2d_145/PartitionedCallÊ
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_145/PartitionedCall:output:0conv2d_146_169406conv2d_146_169408*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_146_layer_call_and_return_conditional_losses_1690542$
"conv2d_146/StatefulPartitionedCall
!max_pooling2d_146/PartitionedCallPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_1689832#
!max_pooling2d_146/PartitionedCall
"dropout_37/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1691792$
"dropout_37/StatefulPartitionedCall
flatten_48/PartitionedCallPartitionedCall+dropout_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_48_layer_call_and_return_conditional_losses_1690742
flatten_48/PartitionedCall²
 dense_96/StatefulPartitionedCallStatefulPartitionedCall#flatten_48/PartitionedCall:output:0dense_96_169414dense_96_169416*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_1690872"
 dense_96/StatefulPartitionedCall·
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_169419dense_97_169421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_1691032"
 dense_97/StatefulPartitionedCall
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

IdentityÐ
NoOpNoOp#^conv2d_144/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall#^dropout_37/StatefulPartitionedCall&^sequential_88/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : 2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall2H
"dropout_37/StatefulPartitionedCall"dropout_37/StatefulPartitionedCall2N
%sequential_88/StatefulPartitionedCall%sequential_88/StatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
-
_user_specified_namesequential_88_input

X
.__inference_sequential_88_layer_call_fn_168577
random_flip_41_input
identityß
PartitionedCallPartitionedCallrandom_flip_41_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1685742
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:g c
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
.
_user_specified_namerandom_flip_41_input
¦
 
+__inference_conv2d_145_layer_call_fn_170226

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_145_layer_call_and_return_conditional_losses_1690362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿúú: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú
 
_user_specified_nameinputs
õ
Ý
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
	unknown_8:
	unknown_9:	

unknown_10:	


unknown_11:

identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_89_layer_call_and_return_conditional_losses_1692872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
7:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
£
e
I__inference_sequential_88_layer_call_and_return_conditional_losses_169887

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
Ñ4

I__inference_sequential_89_layer_call_and_return_conditional_losses_169383
sequential_88_input+
conv2d_144_169352:
conv2d_144_169354:+
conv2d_145_169358: 
conv2d_145_169360: +
conv2d_146_169364: @
conv2d_146_169366:@$
dense_96_169372:
dense_96_169374:	"
dense_97_169377:	

dense_97_169379:

identity¢"conv2d_144/StatefulPartitionedCall¢"conv2d_145/StatefulPartitionedCall¢"conv2d_146/StatefulPartitionedCall¢ dense_96/StatefulPartitionedCall¢ dense_97/StatefulPartitionedCallú
sequential_88/PartitionedCallPartitionedCallsequential_88_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1685742
sequential_88/PartitionedCall
rescaling_51/PartitionedCallPartitionedCall&sequential_88/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_rescaling_51_layer_call_and_return_conditional_losses_1690052
rescaling_51/PartitionedCallÇ
"conv2d_144/StatefulPartitionedCallStatefulPartitionedCall%rescaling_51/PartitionedCall:output:0conv2d_144_169352conv2d_144_169354*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_144_layer_call_and_return_conditional_losses_1690182$
"conv2d_144/StatefulPartitionedCall
!max_pooling2d_144/PartitionedCallPartitionedCall+conv2d_144/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_1689592#
!max_pooling2d_144/PartitionedCallÌ
"conv2d_145/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_144/PartitionedCall:output:0conv2d_145_169358conv2d_145_169360*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_145_layer_call_and_return_conditional_losses_1690362$
"conv2d_145/StatefulPartitionedCall
!max_pooling2d_145/PartitionedCallPartitionedCall+conv2d_145/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}} * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_1689712#
!max_pooling2d_145/PartitionedCallÊ
"conv2d_146/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_145/PartitionedCall:output:0conv2d_146_169364conv2d_146_169366*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_146_layer_call_and_return_conditional_losses_1690542$
"conv2d_146/StatefulPartitionedCall
!max_pooling2d_146/PartitionedCallPartitionedCall+conv2d_146/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_1689832#
!max_pooling2d_146/PartitionedCall
dropout_37/PartitionedCallPartitionedCall*max_pooling2d_146/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_37_layer_call_and_return_conditional_losses_1690662
dropout_37/PartitionedCallù
flatten_48/PartitionedCallPartitionedCall#dropout_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_48_layer_call_and_return_conditional_losses_1690742
flatten_48/PartitionedCall²
 dense_96/StatefulPartitionedCallStatefulPartitionedCall#flatten_48/PartitionedCall:output:0dense_96_169372dense_96_169374*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_96_layer_call_and_return_conditional_losses_1690872"
 dense_96/StatefulPartitionedCall·
 dense_97/StatefulPartitionedCallStatefulPartitionedCall)dense_96/StatefulPartitionedCall:output:0dense_97_169377dense_97_169379*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_97_layer_call_and_return_conditional_losses_1691032"
 dense_97/StatefulPartitionedCall
IdentityIdentity)dense_97/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity
NoOpNoOp#^conv2d_144/StatefulPartitionedCall#^conv2d_145/StatefulPartitionedCall#^conv2d_146/StatefulPartitionedCall!^dense_96/StatefulPartitionedCall!^dense_97/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : 2H
"conv2d_144/StatefulPartitionedCall"conv2d_144/StatefulPartitionedCall2H
"conv2d_145/StatefulPartitionedCall"conv2d_145/StatefulPartitionedCall2H
"conv2d_146/StatefulPartitionedCall"conv2d_146/StatefulPartitionedCall2D
 dense_96/StatefulPartitionedCall dense_96/StatefulPartitionedCall2D
 dense_97/StatefulPartitionedCall dense_97/StatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
-
_user_specified_namesequential_88_input
ö

/__inference_random_flip_41_layer_call_fn_170498

inputs
unknown:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1688882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿôô: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ø
ÿ
F__inference_conv2d_144_layer_call_and_return_conditional_losses_170197

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿôô: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
¨
j
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170502

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ì
ÿ
F__inference_conv2d_146_layer_call_and_return_conditional_losses_170237

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ}} : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}} 
 
_user_specified_nameinputs
ª
d
H__inference_rescaling_51_layer_call_and_return_conditional_losses_169005

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
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
:ÿÿÿÿÿÿÿÿÿôô2
mulk
addAddV2mul:z:0Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
adde
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
A
«
I__inference_sequential_89_layer_call_and_return_conditional_losses_169506

inputsC
)conv2d_144_conv2d_readvariableop_resource:8
*conv2d_144_biasadd_readvariableop_resource:C
)conv2d_145_conv2d_readvariableop_resource: 8
*conv2d_145_biasadd_readvariableop_resource: C
)conv2d_146_conv2d_readvariableop_resource: @8
*conv2d_146_biasadd_readvariableop_resource:@<
'dense_96_matmul_readvariableop_resource:7
(dense_96_biasadd_readvariableop_resource:	:
'dense_97_matmul_readvariableop_resource:	
6
(dense_97_biasadd_readvariableop_resource:

identity¢!conv2d_144/BiasAdd/ReadVariableOp¢ conv2d_144/Conv2D/ReadVariableOp¢!conv2d_145/BiasAdd/ReadVariableOp¢ conv2d_145/Conv2D/ReadVariableOp¢!conv2d_146/BiasAdd/ReadVariableOp¢ conv2d_146/Conv2D/ReadVariableOp¢dense_96/BiasAdd/ReadVariableOp¢dense_96/MatMul/ReadVariableOp¢dense_97/BiasAdd/ReadVariableOp¢dense_97/MatMul/ReadVariableOpo
rescaling_51/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_51/Cast/xs
rescaling_51/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_51/Cast_1/x
rescaling_51/mulMulinputsrescaling_51/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
rescaling_51/mul
rescaling_51/addAddV2rescaling_51/mul:z:0rescaling_51/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
rescaling_51/add¶
 conv2d_144/Conv2D/ReadVariableOpReadVariableOp)conv2d_144_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02"
 conv2d_144/Conv2D/ReadVariableOpÔ
conv2d_144/Conv2DConv2Drescaling_51/add:z:0(conv2d_144/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
paddingSAME*
strides
2
conv2d_144/Conv2D­
!conv2d_144/BiasAdd/ReadVariableOpReadVariableOp*conv2d_144_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!conv2d_144/BiasAdd/ReadVariableOp¶
conv2d_144/BiasAddBiasAddconv2d_144/Conv2D:output:0)conv2d_144/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
conv2d_144/BiasAdd
conv2d_144/ReluReluconv2d_144/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2
conv2d_144/ReluÏ
max_pooling2d_144/MaxPoolMaxPoolconv2d_144/Relu:activations:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú*
ksize
*
paddingVALID*
strides
2
max_pooling2d_144/MaxPool¶
 conv2d_145/Conv2D/ReadVariableOpReadVariableOp)conv2d_145_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 conv2d_145/Conv2D/ReadVariableOpâ
conv2d_145/Conv2DConv2D"max_pooling2d_144/MaxPool:output:0(conv2d_145/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú *
paddingSAME*
strides
2
conv2d_145/Conv2D­
!conv2d_145/BiasAdd/ReadVariableOpReadVariableOp*conv2d_145_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_145/BiasAdd/ReadVariableOp¶
conv2d_145/BiasAddBiasAddconv2d_145/Conv2D:output:0)conv2d_145/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2
conv2d_145/BiasAdd
conv2d_145/ReluReluconv2d_145/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿúú 2
conv2d_145/ReluÍ
max_pooling2d_145/MaxPoolMaxPoolconv2d_145/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}} *
ksize
*
paddingVALID*
strides
2
max_pooling2d_145/MaxPool¶
 conv2d_146/Conv2D/ReadVariableOpReadVariableOp)conv2d_146_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02"
 conv2d_146/Conv2D/ReadVariableOpà
conv2d_146/Conv2DConv2D"max_pooling2d_145/MaxPool:output:0(conv2d_146/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingSAME*
strides
2
conv2d_146/Conv2D­
!conv2d_146/BiasAdd/ReadVariableOpReadVariableOp*conv2d_146_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_146/BiasAdd/ReadVariableOp´
conv2d_146/BiasAddBiasAddconv2d_146/Conv2D:output:0)conv2d_146/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_146/BiasAdd
conv2d_146/ReluReluconv2d_146/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_146/ReluÍ
max_pooling2d_146/MaxPoolMaxPoolconv2d_146/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_146/MaxPool
dropout_37/IdentityIdentity"max_pooling2d_146/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2
dropout_37/Identityu
flatten_48/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ Á 2
flatten_48/Const 
flatten_48/ReshapeReshapedropout_37/Identity:output:0flatten_48/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_48/Reshape«
dense_96/MatMul/ReadVariableOpReadVariableOp'dense_96_matmul_readvariableop_resource*!
_output_shapes
:*
dtype02 
dense_96/MatMul/ReadVariableOp¤
dense_96/MatMulMatMulflatten_48/Reshape:output:0&dense_96/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_96/MatMul¨
dense_96/BiasAdd/ReadVariableOpReadVariableOp(dense_96_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_96/BiasAdd/ReadVariableOp¦
dense_96/BiasAddBiasAdddense_96/MatMul:product:0'dense_96/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_96/BiasAddt
dense_96/ReluReludense_96/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_96/Relu©
dense_97/MatMul/ReadVariableOpReadVariableOp'dense_97_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02 
dense_97/MatMul/ReadVariableOp£
dense_97/MatMulMatMuldense_96/Relu:activations:0&dense_97/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_97/MatMul§
dense_97/BiasAdd/ReadVariableOpReadVariableOp(dense_97_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_97/BiasAdd/ReadVariableOp¥
dense_97/BiasAddBiasAdddense_97/MatMul:product:0'dense_97/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_97/BiasAddt
IdentityIdentitydense_97/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity©
NoOpNoOp"^conv2d_144/BiasAdd/ReadVariableOp!^conv2d_144/Conv2D/ReadVariableOp"^conv2d_145/BiasAdd/ReadVariableOp!^conv2d_145/Conv2D/ReadVariableOp"^conv2d_146/BiasAdd/ReadVariableOp!^conv2d_146/Conv2D/ReadVariableOp ^dense_96/BiasAdd/ReadVariableOp^dense_96/MatMul/ReadVariableOp ^dense_97/BiasAdd/ReadVariableOp^dense_97/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : 2F
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
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
î
J
.__inference_sequential_88_layer_call_fn_170162

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_88_layer_call_and_return_conditional_losses_1685742
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ï
f
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170327

inputs
identity}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
f
J__inference_random_flip_41_layer_call_and_return_conditional_losses_168559

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿôô:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
í
Ú
I__inference_sequential_88_layer_call_and_return_conditional_losses_168913

inputs#
random_flip_41_168903:	'
random_rotation_40_168906:	#
random_zoom_40_168909:	
identity¢&random_flip_41/StatefulPartitionedCall¢*random_rotation_40/StatefulPartitionedCall¢&random_zoom_40/StatefulPartitionedCall 
&random_flip_41/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_flip_41_168903*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1688882(
&random_flip_41/StatefulPartitionedCallÙ
*random_rotation_40/StatefulPartitionedCallStatefulPartitionedCall/random_flip_41/StatefulPartitionedCall:output:0random_rotation_40_168906*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1688182,
*random_rotation_40/StatefulPartitionedCallÍ
&random_zoom_40/StatefulPartitionedCallStatefulPartitionedCall3random_rotation_40/StatefulPartitionedCall:output:0random_zoom_40_168909*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1686872(
&random_zoom_40/StatefulPartitionedCall
IdentityIdentity/random_zoom_40/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

IdentityÍ
NoOpNoOp'^random_flip_41/StatefulPartitionedCall+^random_rotation_40/StatefulPartitionedCall'^random_zoom_40/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿôô: : : 2P
&random_flip_41/StatefulPartitionedCall&random_flip_41/StatefulPartitionedCall2X
*random_rotation_40/StatefulPartitionedCall*random_rotation_40/StatefulPartitionedCall2P
&random_zoom_40/StatefulPartitionedCall&random_zoom_40/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
äf

J__inference_random_flip_41_layer_call_and_return_conditional_losses_170474

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkip¢Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shape
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Const½
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prod
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/x¥
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkip¨
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stack¬
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1¬
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice´
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcast¬
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack°
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1°
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2ü
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1º
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg®
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

zeros_like
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
strided_slice/stack_2
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceÕ
3stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô25
3stateless_random_flip_left_right/control_dependency¼
&stateless_random_flip_left_right/ShapeShape<stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/Shape¶
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4stateless_random_flip_left_right/strided_slice/stackº
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_1º
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2¨
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.stateless_random_flip_left_right/strided_sliceñ
?stateless_random_flip_left_right/stateless_random_uniform/shapePack7stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2A
?stateless_random_flip_left_right/stateless_random_uniform/shapeÃ
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/minÃ
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2?
=stateless_random_flip_left_right/stateless_random_uniform/max
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¬
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgÊ
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2¶
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subÓ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=stateless_random_flip_left_right/stateless_random_uniform/mul¸
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9stateless_random_flip_left_right/stateless_random_uniform¦
0stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/1¦
0stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/2¦
0stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :22
0stateless_random_flip_left_right/Reshape/shape/3
.stateless_random_flip_left_right/Reshape/shapePack7stateless_random_flip_left_right/strided_slice:output:09stateless_random_flip_left_right/Reshape/shape/1:output:09stateless_random_flip_left_right/Reshape/shape/2:output:09stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.stateless_random_flip_left_right/Reshape/shape
(stateless_random_flip_left_right/ReshapeReshape=stateless_random_flip_left_right/stateless_random_uniform:z:07stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(stateless_random_flip_left_right/ReshapeÆ
&stateless_random_flip_left_right/RoundRound1stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_left_right/Round¬
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axis
*stateless_random_flip_left_right/ReverseV2	ReverseV2<stateless_random_flip_left_right/control_dependency:output:08stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2,
*stateless_random_flip_left_right/ReverseV2ð
$stateless_random_flip_left_right/mulMul*stateless_random_flip_left_right/Round:y:03stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2&
$stateless_random_flip_left_right/mul
&stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&stateless_random_flip_left_right/sub/xê
$stateless_random_flip_left_right/subSub/stateless_random_flip_left_right/sub/x:output:0*stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$stateless_random_flip_left_right/subû
&stateless_random_flip_left_right/mul_1Mul(stateless_random_flip_left_right/sub:z:0<stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2(
&stateless_random_flip_left_right/mul_1ç
$stateless_random_flip_left_right/addAddV2(stateless_random_flip_left_right/mul:z:0*stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2&
$stateless_random_flip_left_right/add
IdentityIdentity(stateless_random_flip_left_right/add:z:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identity¤
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkipP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿôô: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2¢
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
­
i
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_168959

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ê
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
	unknown_8:
	unknown_9:	

unknown_10:	


unknown_11:

identity¢StatefulPartitionedCall
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
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_89_layer_call_and_return_conditional_losses_1692872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
7:ÿÿÿÿÿÿÿÿÿôô: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
-
_user_specified_namesequential_88_input
­
i
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_168983

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
Ç
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170620

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkipD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
stateful_uniform/Const
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
stateful_uniform/Cast/x
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1Ù
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkip
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stack
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2Î
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_slice
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stack
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2Æ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1 
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg¸
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateful_uniform/StatelessRandomUniformV2
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub¯
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_1/y
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_2/y
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yª
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_5/y
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_6/y
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_7/y
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y°
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2Â
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_2
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack£
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1£
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2÷
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_2
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack£
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1£
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2÷
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Neg
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack£
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1£
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2ù
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_3
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack£
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1£
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2÷
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_3
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack£
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1£
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2÷
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack£
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1£
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2û
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1Ã
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
rotation_matrix/zeros/Constµ
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis¨
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
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
transform/fill_valueÉ
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3
IdentityIdentity9transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿôô: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
ö

/__inference_random_zoom_40_layer_call_fn_170750

inputs
unknown:	
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_1686872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿôô: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
Ú

/__inference_random_flip_41_layer_call_fn_170486

inputs
unknown:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_random_flip_41_layer_call_and_return_conditional_losses_1684482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ

3__inference_random_rotation_40_layer_call_fn_170632

inputs
unknown:	
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_1688182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿôô: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿôô
 
_user_specified_nameinputs
Û
N
2__inference_max_pooling2d_145_layer_call_fn_168977

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_1689712
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Í
serving_default¹
]
sequential_88_inputF
%serving_default_sequential_88_input:0ÿÿÿÿÿÿÿÿÿôô<
dense_970
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:´

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
Ö_default_save_signature
+×&call_and_return_all_conditional_losses
Ø__call__"
_tf_keras_sequential
Ó
layer-0
layer-1
layer-2
trainable_variables
	variables
regularization_losses
	keras_api
+Ù&call_and_return_all_conditional_losses
Ú__call__"
_tf_keras_sequential
§
trainable_variables
	variables
regularization_losses
	keras_api
+Û&call_and_return_all_conditional_losses
Ü__call__"
_tf_keras_layer
½

kernel
bias
 trainable_variables
!	variables
"regularization_losses
#	keras_api
+Ý&call_and_return_all_conditional_losses
Þ__call__"
_tf_keras_layer
§
$trainable_variables
%	variables
&regularization_losses
'	keras_api
+ß&call_and_return_all_conditional_losses
à__call__"
_tf_keras_layer
½

(kernel
)bias
*trainable_variables
+	variables
,regularization_losses
-	keras_api
+á&call_and_return_all_conditional_losses
â__call__"
_tf_keras_layer
§
.trainable_variables
/	variables
0regularization_losses
1	keras_api
+ã&call_and_return_all_conditional_losses
ä__call__"
_tf_keras_layer
½

2kernel
3bias
4trainable_variables
5	variables
6regularization_losses
7	keras_api
+å&call_and_return_all_conditional_losses
æ__call__"
_tf_keras_layer
§
8trainable_variables
9	variables
:regularization_losses
;	keras_api
+ç&call_and_return_all_conditional_losses
è__call__"
_tf_keras_layer
§
<trainable_variables
=	variables
>regularization_losses
?	keras_api
+é&call_and_return_all_conditional_losses
ê__call__"
_tf_keras_layer
§
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
+ë&call_and_return_all_conditional_losses
ì__call__"
_tf_keras_layer
½

Dkernel
Ebias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+í&call_and_return_all_conditional_losses
î__call__"
_tf_keras_layer
½

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
+ï&call_and_return_all_conditional_losses
ð__call__"
_tf_keras_layer
©
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_ratemÂmÃ(mÄ)mÅ2mÆ3mÇDmÈEmÉJmÊKmËvÌvÍ(vÎ)vÏ2vÐ3vÑDvÒEvÓJvÔKvÕ"
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
Î
Ulayer_metrics
trainable_variables
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
	variables
regularization_losses
Ymetrics
Ø__call__
Ö_default_save_signature
+×&call_and_return_all_conditional_losses
'×"call_and_return_conditional_losses"
_generic_user_object
-
ñserving_default"
signature_map
±
Z_rng
[trainable_variables
\	variables
]regularization_losses
^	keras_api
+ò&call_and_return_all_conditional_losses
ó__call__"
_tf_keras_layer
±
__rng
`trainable_variables
a	variables
bregularization_losses
c	keras_api
+ô&call_and_return_all_conditional_losses
õ__call__"
_tf_keras_layer
±
d_rng
etrainable_variables
f	variables
gregularization_losses
h	keras_api
+ö&call_and_return_all_conditional_losses
÷__call__"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
ilayer_metrics
trainable_variables
jlayer_regularization_losses

klayers
lnon_trainable_variables
	variables
regularization_losses
mmetrics
Ú__call__
+Ù&call_and_return_all_conditional_losses
'Ù"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
nlayer_metrics
olayer_regularization_losses
trainable_variables

players
qnon_trainable_variables
	variables
regularization_losses
rmetrics
Ü__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
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
°
slayer_metrics
tlayer_regularization_losses
 trainable_variables

ulayers
vnon_trainable_variables
!	variables
"regularization_losses
wmetrics
Þ__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
xlayer_metrics
ylayer_regularization_losses
$trainable_variables

zlayers
{non_trainable_variables
%	variables
&regularization_losses
|metrics
à__call__
+ß&call_and_return_all_conditional_losses
'ß"call_and_return_conditional_losses"
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
²
}layer_metrics
~layer_regularization_losses
*trainable_variables

layers
non_trainable_variables
+	variables
,regularization_losses
metrics
â__call__
+á&call_and_return_all_conditional_losses
'á"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
 layer_regularization_losses
.trainable_variables
layers
non_trainable_variables
/	variables
0regularization_losses
metrics
ä__call__
+ã&call_and_return_all_conditional_losses
'ã"call_and_return_conditional_losses"
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
µ
layer_metrics
 layer_regularization_losses
4trainable_variables
layers
non_trainable_variables
5	variables
6regularization_losses
metrics
æ__call__
+å&call_and_return_all_conditional_losses
'å"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
 layer_regularization_losses
8trainable_variables
layers
non_trainable_variables
9	variables
:regularization_losses
metrics
è__call__
+ç&call_and_return_all_conditional_losses
'ç"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
 layer_regularization_losses
<trainable_variables
layers
non_trainable_variables
=	variables
>regularization_losses
metrics
ê__call__
+é&call_and_return_all_conditional_losses
'é"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layer_metrics
 layer_regularization_losses
@trainable_variables
layers
non_trainable_variables
A	variables
Bregularization_losses
metrics
ì__call__
+ë&call_and_return_all_conditional_losses
'ë"call_and_return_conditional_losses"
_generic_user_object
$:"2dense_96/kernel
:2dense_96/bias
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
µ
layer_metrics
 layer_regularization_losses
Ftrainable_variables
layers
non_trainable_variables
G	variables
Hregularization_losses
metrics
î__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
": 	
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
µ
 layer_metrics
 ¡layer_regularization_losses
Ltrainable_variables
¢layers
£non_trainable_variables
M	variables
Nregularization_losses
¤metrics
ð__call__
+ï&call_and_return_all_conditional_losses
'ï"call_and_return_conditional_losses"
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
¥0
¦1"
trackable_list_wrapper
/
§
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¨layer_metrics
 ©layer_regularization_losses
[trainable_variables
ªlayers
«non_trainable_variables
\	variables
]regularization_losses
¬metrics
ó__call__
+ò&call_and_return_all_conditional_losses
'ò"call_and_return_conditional_losses"
_generic_user_object
/
­
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
®layer_metrics
 ¯layer_regularization_losses
`trainable_variables
°layers
±non_trainable_variables
a	variables
bregularization_losses
²metrics
õ__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
/
³
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
´layer_metrics
 µlayer_regularization_losses
etrainable_variables
¶layers
·non_trainable_variables
f	variables
gregularization_losses
¸metrics
÷__call__
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
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

¹total

ºcount
»	variables
¼	keras_api"
_tf_keras_metric
c

½total

¾count
¿
_fn_kwargs
À	variables
Á	keras_api"
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
¹0
º1"
trackable_list_wrapper
.
»	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
½0
¾1"
trackable_list_wrapper
.
À	variables"
_generic_user_object
0:.2Adam/conv2d_144/kernel/m
": 2Adam/conv2d_144/bias/m
0:. 2Adam/conv2d_145/kernel/m
":  2Adam/conv2d_145/bias/m
0:. @2Adam/conv2d_146/kernel/m
": @2Adam/conv2d_146/bias/m
):'2Adam/dense_96/kernel/m
!:2Adam/dense_96/bias/m
':%	
2Adam/dense_97/kernel/m
 :
2Adam/dense_97/bias/m
0:.2Adam/conv2d_144/kernel/v
": 2Adam/conv2d_144/bias/v
0:. 2Adam/conv2d_145/kernel/v
":  2Adam/conv2d_145/bias/v
0:. @2Adam/conv2d_146/kernel/v
": @2Adam/conv2d_146/bias/v
):'2Adam/dense_96/kernel/v
!:2Adam/dense_96/bias/v
':%	
2Adam/dense_97/kernel/v
 :
2Adam/dense_97/bias/v
õ2ò
!__inference__wrapped_model_168345Ì
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *<¢9
74
sequential_88_inputÿÿÿÿÿÿÿÿÿôô
ò2ï
I__inference_sequential_89_layer_call_and_return_conditional_losses_169506
I__inference_sequential_89_layer_call_and_return_conditional_losses_169827
I__inference_sequential_89_layer_call_and_return_conditional_losses_169383
I__inference_sequential_89_layer_call_and_return_conditional_losses_169425À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_sequential_89_layer_call_fn_169133
.__inference_sequential_89_layer_call_fn_169852
.__inference_sequential_89_layer_call_fn_169883
.__inference_sequential_89_layer_call_fn_169347À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_88_layer_call_and_return_conditional_losses_169887
I__inference_sequential_88_layer_call_and_return_conditional_losses_170157
I__inference_sequential_88_layer_call_and_return_conditional_losses_168940
I__inference_sequential_88_layer_call_and_return_conditional_losses_168953À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
.__inference_sequential_88_layer_call_fn_168577
.__inference_sequential_88_layer_call_fn_170162
.__inference_sequential_88_layer_call_fn_170173
.__inference_sequential_88_layer_call_fn_168933À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
H__inference_rescaling_51_layer_call_and_return_conditional_losses_170181¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_rescaling_51_layer_call_fn_170186¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_conv2d_144_layer_call_and_return_conditional_losses_170197¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_conv2d_144_layer_call_fn_170206¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_168959à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
2__inference_max_pooling2d_144_layer_call_fn_168965à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ð2í
F__inference_conv2d_145_layer_call_and_return_conditional_losses_170217¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_conv2d_145_layer_call_fn_170226¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_168971à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
2__inference_max_pooling2d_145_layer_call_fn_168977à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ð2í
F__inference_conv2d_146_layer_call_and_return_conditional_losses_170237¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_conv2d_146_layer_call_fn_170246¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
µ2²
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_168983à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
2__inference_max_pooling2d_146_layer_call_fn_168989à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ê2Ç
F__inference_dropout_37_layer_call_and_return_conditional_losses_170251
F__inference_dropout_37_layer_call_and_return_conditional_losses_170263´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_dropout_37_layer_call_fn_170268
+__inference_dropout_37_layer_call_fn_170273´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_flatten_48_layer_call_and_return_conditional_losses_170279¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_flatten_48_layer_call_fn_170284¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_96_layer_call_and_return_conditional_losses_170295¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_96_layer_call_fn_170304¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_97_layer_call_and_return_conditional_losses_170314¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_97_layer_call_fn_170323¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×BÔ
$__inference_signature_wrapper_169458sequential_88_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170327
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170412
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170416
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170474´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
þ2û
/__inference_random_flip_41_layer_call_fn_170479
/__inference_random_flip_41_layer_call_fn_170486
/__inference_random_flip_41_layer_call_fn_170491
/__inference_random_flip_41_layer_call_fn_170498´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170502
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170620´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¤2¡
3__inference_random_rotation_40_layer_call_fn_170625
3__inference_random_rotation_40_layer_call_fn_170632´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170636
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170738´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
/__inference_random_zoom_40_layer_call_fn_170743
/__inference_random_zoom_40_layer_call_fn_170750´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ¯
!__inference__wrapped_model_168345
()23DEJKF¢C
<¢9
74
sequential_88_inputÿÿÿÿÿÿÿÿÿôô
ª "3ª0
.
dense_97"
dense_97ÿÿÿÿÿÿÿÿÿ
º
F__inference_conv2d_144_layer_call_and_return_conditional_losses_170197p9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿôô
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 
+__inference_conv2d_144_layer_call_fn_170206c9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿôô
ª ""ÿÿÿÿÿÿÿÿÿôôº
F__inference_conv2d_145_layer_call_and_return_conditional_losses_170217p()9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿúú
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿúú 
 
+__inference_conv2d_145_layer_call_fn_170226c()9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿúú
ª ""ÿÿÿÿÿÿÿÿÿúú ¶
F__inference_conv2d_146_layer_call_and_return_conditional_losses_170237l237¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ}} 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ}}@
 
+__inference_conv2d_146_layer_call_fn_170246_237¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ}} 
ª " ÿÿÿÿÿÿÿÿÿ}}@§
D__inference_dense_96_layer_call_and_return_conditional_losses_170295_DE1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_dense_96_layer_call_fn_170304RDE1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_dense_97_layer_call_and_return_conditional_losses_170314]JK0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 }
)__inference_dense_97_layer_call_fn_170323PJK0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
¶
F__inference_dropout_37_layer_call_and_return_conditional_losses_170251l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ>>@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ>>@
 ¶
F__inference_dropout_37_layer_call_and_return_conditional_losses_170263l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ>>@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ>>@
 
+__inference_dropout_37_layer_call_fn_170268_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ>>@
p 
ª " ÿÿÿÿÿÿÿÿÿ>>@
+__inference_dropout_37_layer_call_fn_170273_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ>>@
p
ª " ÿÿÿÿÿÿÿÿÿ>>@¬
F__inference_flatten_48_layer_call_and_return_conditional_losses_170279b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ>>@
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_flatten_48_layer_call_fn_170284U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ>>@
ª "ÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_144_layer_call_and_return_conditional_losses_168959R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_144_layer_call_fn_168965R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_145_layer_call_and_return_conditional_losses_168971R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_145_layer_call_fn_168977R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_146_layer_call_and_return_conditional_losses_168983R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_146_layer_call_fn_168989R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿñ
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170327¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 õ
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170412¦§V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170416p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 Â
J__inference_random_flip_41_layer_call_and_return_conditional_losses_170474t§=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 É
/__inference_random_flip_41_layer_call_fn_170479V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÍ
/__inference_random_flip_41_layer_call_fn_170486§V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
/__inference_random_flip_41_layer_call_fn_170491c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p 
ª ""ÿÿÿÿÿÿÿÿÿôô
/__inference_random_flip_41_layer_call_fn_170498g§=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p
ª ""ÿÿÿÿÿÿÿÿÿôôÂ
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170502p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 Æ
N__inference_random_rotation_40_layer_call_and_return_conditional_losses_170620t­=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 
3__inference_random_rotation_40_layer_call_fn_170625c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p 
ª ""ÿÿÿÿÿÿÿÿÿôô
3__inference_random_rotation_40_layer_call_fn_170632g­=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p
ª ""ÿÿÿÿÿÿÿÿÿôô¾
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170636p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 Â
J__inference_random_zoom_40_layer_call_and_return_conditional_losses_170738t³=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 
/__inference_random_zoom_40_layer_call_fn_170743c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p 
ª ""ÿÿÿÿÿÿÿÿÿôô
/__inference_random_zoom_40_layer_call_fn_170750g³=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿôô
p
ª ""ÿÿÿÿÿÿÿÿÿôô¸
H__inference_rescaling_51_layer_call_and_return_conditional_losses_170181l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿôô
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 
-__inference_rescaling_51_layer_call_fn_170186_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿôô
ª ""ÿÿÿÿÿÿÿÿÿôôÐ
I__inference_sequential_88_layer_call_and_return_conditional_losses_168940O¢L
E¢B
85
random_flip_41_inputÿÿÿÿÿÿÿÿÿôô
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 Ø
I__inference_sequential_88_layer_call_and_return_conditional_losses_168953§­³O¢L
E¢B
85
random_flip_41_inputÿÿÿÿÿÿÿÿÿôô
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 Á
I__inference_sequential_88_layer_call_and_return_conditional_losses_169887tA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôô
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 É
I__inference_sequential_88_layer_call_and_return_conditional_losses_170157|§­³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôô
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿôô
 §
.__inference_sequential_88_layer_call_fn_168577uO¢L
E¢B
85
random_flip_41_inputÿÿÿÿÿÿÿÿÿôô
p 

 
ª ""ÿÿÿÿÿÿÿÿÿôô¯
.__inference_sequential_88_layer_call_fn_168933}§­³O¢L
E¢B
85
random_flip_41_inputÿÿÿÿÿÿÿÿÿôô
p

 
ª ""ÿÿÿÿÿÿÿÿÿôô
.__inference_sequential_88_layer_call_fn_170162gA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôô
p 

 
ª ""ÿÿÿÿÿÿÿÿÿôô¡
.__inference_sequential_88_layer_call_fn_170173o§­³A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôô
p

 
ª ""ÿÿÿÿÿÿÿÿÿôôÑ
I__inference_sequential_89_layer_call_and_return_conditional_losses_169383
()23DEJKN¢K
D¢A
74
sequential_88_inputÿÿÿÿÿÿÿÿÿôô
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ×
I__inference_sequential_89_layer_call_and_return_conditional_losses_169425§­³()23DEJKN¢K
D¢A
74
sequential_88_inputÿÿÿÿÿÿÿÿÿôô
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ã
I__inference_sequential_89_layer_call_and_return_conditional_losses_169506v
()23DEJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôô
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 É
I__inference_sequential_89_layer_call_and_return_conditional_losses_169827|§­³()23DEJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôô
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¨
.__inference_sequential_89_layer_call_fn_169133v
()23DEJKN¢K
D¢A
74
sequential_88_inputÿÿÿÿÿÿÿÿÿôô
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
®
.__inference_sequential_89_layer_call_fn_169347|§­³()23DEJKN¢K
D¢A
74
sequential_88_inputÿÿÿÿÿÿÿÿÿôô
p

 
ª "ÿÿÿÿÿÿÿÿÿ

.__inference_sequential_89_layer_call_fn_169852i
()23DEJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôô
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¡
.__inference_sequential_89_layer_call_fn_169883o§­³()23DEJKA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿôô
p

 
ª "ÿÿÿÿÿÿÿÿÿ
É
$__inference_signature_wrapper_169458 
()23DEJK]¢Z
¢ 
SªP
N
sequential_88_input74
sequential_88_inputÿÿÿÿÿÿÿÿÿôô"3ª0
.
dense_97"
dense_97ÿÿÿÿÿÿÿÿÿ
