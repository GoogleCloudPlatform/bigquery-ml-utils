ĶØ
)Ū(
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
”
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
É
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0ž’’’’’’’’"
value_indexint(0ž’’’’’’’’"+

vocab_sizeint’’’’’’’’’(0’’’’’’’’’"
	delimiterstring	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype
2
LookupTableSizeV2
table_handle
size	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
k
NotEqual
x"T
y"T
z
"
Ttype:
2	
"$
incompatible_shape_errorbool(

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint’’’’’’’’’"	
Ttype"
TItype0	:
2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
¼
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized

s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
E
Where

input"T	
index	"%
Ttype0
:
2	
"serve*1.15.52v1.15.4-39-g3db52be8Ę

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 
k
global_step
VariableV2*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: 

global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
T0	*
_class
loc:@global_step*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
f
PlaceholderPlaceholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
h
Placeholder_1Placeholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
h
Placeholder_2Placeholder*#
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’

Placeholder_3Placeholder*0
_output_shapes
:’’’’’’’’’’’’’’’’’’*
dtype0*%
shape:’’’’’’’’’’’’’’’’’’

]dnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 

Wdnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/NotEqualNotEqualPlaceholder_3]dnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/ignore_value/x*
T0*0
_output_shapes
:’’’’’’’’’’’’’’’’’’
į
Vdnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/indicesWhereWdnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/NotEqual*'
_output_shapes
:’’’’’’’’’

Udnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/valuesGatherNdPlaceholder_3Vdnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:’’’’’’’’’
§
Zdnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/dense_shapeShapePlaceholder_3*
T0*
_output_shapes
:*
out_type0	

ednn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/asset_pathConst"/device:CPU:**
_output_shapes
: *
dtype0*\
valueSBQ BKgs://bqml_dnn_classifier_13223904274911629959_eoid/vocabulary/array_col.csv
«
`dnn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
’’’’’’’’’
”
ednn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*o
shared_name`^hash_table_gs://bqml_dnn_classifier_13223904274911629959_eoid/vocabulary/array_col.csv_8_-2_-1*
value_dtype0	
»
dnn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/table_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2ednn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/hash_tableednn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/asset_path*
	key_indexž’’’’’’’’*
value_index’’’’’’’’’*

vocab_size

\dnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/hash_bucketStringToHashBucketFastUdnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/values*#
_output_shapes
:’’’’’’’’’*
num_buckets
å
tdnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2ednn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/hash_tableUdnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/values`dnn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:’’’’’’’’’

rdnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/hash_table_Size/LookupTableSizeV2LookupTableSizeV2ednn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/hash_table*
_output_shapes
: 
Ū
Tdnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/AddAdd\dnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/hash_bucketrdnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/hash_table_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:’’’’’’’’’
ė
Ydnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/NotEqualNotEqualtdnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2`dnn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/Const*
T0	*#
_output_shapes
:’’’’’’’’’
ŗ
Ydnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/SelectV2SelectV2Ydnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/NotEqualtdnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2Tdnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/Add*
T0	*#
_output_shapes
:’’’’’’’’’
„
Zdnn/input_from_feature_columns/input_layer/array_col_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
’’’’’’’’’

Ldnn/input_from_feature_columns/input_layer/array_col_indicator/SparseToDenseSparseToDenseVdnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/indicesZdnn/input_from_feature_columns/input_layer/array_col_indicator/to_sparse_input/dense_shapeYdnn/input_from_feature_columns/input_layer/array_col_indicator/hash_table_Lookup/SelectV2Zdnn/input_from_feature_columns/input_layer/array_col_indicator/SparseToDense/default_value*
T0	*
Tindices0	*0
_output_shapes
:’’’’’’’’’’’’’’’’’’

Ldnn/input_from_feature_columns/input_layer/array_col_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

Ndnn/input_from_feature_columns/input_layer/array_col_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    

Ldnn/input_from_feature_columns/input_layer/array_col_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :	

Odnn/input_from_feature_columns/input_layer/array_col_indicator/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

Pdnn/input_from_feature_columns/input_layer/array_col_indicator/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ī
Fdnn/input_from_feature_columns/input_layer/array_col_indicator/one_hotOneHotLdnn/input_from_feature_columns/input_layer/array_col_indicator/SparseToDenseLdnn/input_from_feature_columns/input_layer/array_col_indicator/one_hot/depthOdnn/input_from_feature_columns/input_layer/array_col_indicator/one_hot/on_valuePdnn/input_from_feature_columns/input_layer/array_col_indicator/one_hot/off_value*
T0*4
_output_shapes"
 :’’’’’’’’’’’’’’’’’’	
§
Tdnn/input_from_feature_columns/input_layer/array_col_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ž’’’’’’’’

Bdnn/input_from_feature_columns/input_layer/array_col_indicator/SumSumFdnn/input_from_feature_columns/input_layer/array_col_indicator/one_hotTdnn/input_from_feature_columns/input_layer/array_col_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:’’’’’’’’’	
¶
Ddnn/input_from_feature_columns/input_layer/array_col_indicator/ShapeShapeBdnn/input_from_feature_columns/input_layer/array_col_indicator/Sum*
T0*
_output_shapes
:

Rdnn/input_from_feature_columns/input_layer/array_col_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Tdnn/input_from_feature_columns/input_layer/array_col_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Tdnn/input_from_feature_columns/input_layer/array_col_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
č
Ldnn/input_from_feature_columns/input_layer/array_col_indicator/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/array_col_indicator/ShapeRdnn/input_from_feature_columns/input_layer/array_col_indicator/strided_slice/stackTdnn/input_from_feature_columns/input_layer/array_col_indicator/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/array_col_indicator/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Ndnn/input_from_feature_columns/input_layer/array_col_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	
 
Ldnn/input_from_feature_columns/input_layer/array_col_indicator/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/array_col_indicator/strided_sliceNdnn/input_from_feature_columns/input_layer/array_col_indicator/Reshape/shape/1*
N*
T0*
_output_shapes
:

Fdnn/input_from_feature_columns/input_layer/array_col_indicator/ReshapeReshapeBdnn/input_from_feature_columns/input_layer/array_col_indicator/SumLdnn/input_from_feature_columns/input_layer/array_col_indicator/Reshape/shape*
T0*'
_output_shapes
:’’’’’’’’’	

<dnn/input_from_feature_columns/input_layer/f1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
Ć
8dnn/input_from_feature_columns/input_layer/f1/ExpandDims
ExpandDimsPlaceholder<dnn/input_from_feature_columns/input_layer/f1/ExpandDims/dim*
T0*'
_output_shapes
:’’’’’’’’’
|
3dnn/input_from_feature_columns/input_layer/f1/sub/yConst*
_output_shapes
: *
dtype0*
valueB 23"<]÷ŗ?
Ł
1dnn/input_from_feature_columns/input_layer/f1/subSub8dnn/input_from_feature_columns/input_layer/f1/ExpandDims3dnn/input_from_feature_columns/input_layer/f1/sub/y*
T0*'
_output_shapes
:’’’’’’’’’

7dnn/input_from_feature_columns/input_layer/f1/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2wpfØź@
Ž
5dnn/input_from_feature_columns/input_layer/f1/truedivRealDiv1dnn/input_from_feature_columns/input_layer/f1/sub7dnn/input_from_feature_columns/input_layer/f1/truediv/y*
T0*'
_output_shapes
:’’’’’’’’’
²
2dnn/input_from_feature_columns/input_layer/f1/CastCast5dnn/input_from_feature_columns/input_layer/f1/truediv*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’

3dnn/input_from_feature_columns/input_layer/f1/ShapeShape2dnn/input_from_feature_columns/input_layer/f1/Cast*
T0*
_output_shapes
:

Adnn/input_from_feature_columns/input_layer/f1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Cdnn/input_from_feature_columns/input_layer/f1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Cdnn/input_from_feature_columns/input_layer/f1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

;dnn/input_from_feature_columns/input_layer/f1/strided_sliceStridedSlice3dnn/input_from_feature_columns/input_layer/f1/ShapeAdnn/input_from_feature_columns/input_layer/f1/strided_slice/stackCdnn/input_from_feature_columns/input_layer/f1/strided_slice/stack_1Cdnn/input_from_feature_columns/input_layer/f1/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

=dnn/input_from_feature_columns/input_layer/f1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ķ
;dnn/input_from_feature_columns/input_layer/f1/Reshape/shapePack;dnn/input_from_feature_columns/input_layer/f1/strided_slice=dnn/input_from_feature_columns/input_layer/f1/Reshape/shape/1*
N*
T0*
_output_shapes
:
ć
5dnn/input_from_feature_columns/input_layer/f1/ReshapeReshape2dnn/input_from_feature_columns/input_layer/f1/Cast;dnn/input_from_feature_columns/input_layer/f1/Reshape/shape*
T0*'
_output_shapes
:’’’’’’’’’

<dnn/input_from_feature_columns/input_layer/f2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
Å
8dnn/input_from_feature_columns/input_layer/f2/ExpandDims
ExpandDimsPlaceholder_1<dnn/input_from_feature_columns/input_layer/f2/ExpandDims/dim*
T0*'
_output_shapes
:’’’’’’’’’
|
3dnn/input_from_feature_columns/input_layer/f2/sub/yConst*
_output_shapes
: *
dtype0*
valueB 2"Wšq+9@
Ł
1dnn/input_from_feature_columns/input_layer/f2/subSub8dnn/input_from_feature_columns/input_layer/f2/ExpandDims3dnn/input_from_feature_columns/input_layer/f2/sub/y*
T0*'
_output_shapes
:’’’’’’’’’

7dnn/input_from_feature_columns/input_layer/f2/truediv/yConst*
_output_shapes
: *
dtype0*
valueB 2ŅōvYÅ/@
Ž
5dnn/input_from_feature_columns/input_layer/f2/truedivRealDiv1dnn/input_from_feature_columns/input_layer/f2/sub7dnn/input_from_feature_columns/input_layer/f2/truediv/y*
T0*'
_output_shapes
:’’’’’’’’’
²
2dnn/input_from_feature_columns/input_layer/f2/CastCast5dnn/input_from_feature_columns/input_layer/f2/truediv*

DstT0*

SrcT0*'
_output_shapes
:’’’’’’’’’

3dnn/input_from_feature_columns/input_layer/f2/ShapeShape2dnn/input_from_feature_columns/input_layer/f2/Cast*
T0*
_output_shapes
:

Adnn/input_from_feature_columns/input_layer/f2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Cdnn/input_from_feature_columns/input_layer/f2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Cdnn/input_from_feature_columns/input_layer/f2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

;dnn/input_from_feature_columns/input_layer/f2/strided_sliceStridedSlice3dnn/input_from_feature_columns/input_layer/f2/ShapeAdnn/input_from_feature_columns/input_layer/f2/strided_slice/stackCdnn/input_from_feature_columns/input_layer/f2/strided_slice/stack_1Cdnn/input_from_feature_columns/input_layer/f2/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

=dnn/input_from_feature_columns/input_layer/f2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ķ
;dnn/input_from_feature_columns/input_layer/f2/Reshape/shapePack;dnn/input_from_feature_columns/input_layer/f2/strided_slice=dnn/input_from_feature_columns/input_layer/f2/Reshape/shape/1*
N*
T0*
_output_shapes
:
ć
5dnn/input_from_feature_columns/input_layer/f2/ReshapeReshape2dnn/input_from_feature_columns/input_layer/f2/Cast;dnn/input_from_feature_columns/input_layer/f2/Reshape/shape*
T0*'
_output_shapes
:’’’’’’’’’

Ndnn/input_from_feature_columns/input_layer/string_col_indicator/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
é
Jdnn/input_from_feature_columns/input_layer/string_col_indicator/ExpandDims
ExpandDimsPlaceholder_2Ndnn/input_from_feature_columns/input_layer/string_col_indicator/ExpandDims/dim*
T0*'
_output_shapes
:’’’’’’’’’

^dnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/ignore_value/xConst*
_output_shapes
: *
dtype0*
valueB B 
Ā
Xdnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/NotEqualNotEqualJdnn/input_from_feature_columns/input_layer/string_col_indicator/ExpandDims^dnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:’’’’’’’’’
ć
Wdnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/indicesWhereXdnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/NotEqual*'
_output_shapes
:’’’’’’’’’
Ė
Vdnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/valuesGatherNdJdnn/input_from_feature_columns/input_layer/string_col_indicator/ExpandDimsWdnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:’’’’’’’’’
å
[dnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/dense_shapeShapeJdnn/input_from_feature_columns/input_layer/string_col_indicator/ExpandDims*
T0*
_output_shapes
:*
out_type0	

gdnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/asset_pathConst"/device:CPU:**
_output_shapes
: *
dtype0*]
valueTBR BLgs://bqml_dnn_classifier_13223904274911629959_eoid/vocabulary/string_col.csv
­
bdnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
’’’’’’’’’
¤
gdnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*p
shared_namea_hash_table_gs://bqml_dnn_classifier_13223904274911629959_eoid/vocabulary/string_col.csv_6_-2_-1*
value_dtype0	
Į
dnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/table_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV2gdnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/hash_tablegdnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/asset_path*
	key_indexž’’’’’’’’*
value_index’’’’’’’’’*

vocab_size

]dnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/hash_bucketStringToHashBucketFastVdnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/values*#
_output_shapes
:’’’’’’’’’*
num_buckets
ė
udnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2LookupTableFindV2gdnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/hash_tableVdnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/valuesbdnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/Const*	
Tin0*

Tout0	*#
_output_shapes
:’’’’’’’’’

sdnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/hash_table_Size/LookupTableSizeV2LookupTableSizeV2gdnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/hash_table*
_output_shapes
: 
Ž
Udnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/AddAdd]dnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/hash_bucketsdnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/hash_table_Size/LookupTableSizeV2*
T0	*#
_output_shapes
:’’’’’’’’’
ļ
Zdnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/NotEqualNotEqualudnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2bdnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/Const*
T0	*#
_output_shapes
:’’’’’’’’’
¾
Zdnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/SelectV2SelectV2Zdnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/NotEqualudnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/hash_table_Lookup/LookupTableFindV2Udnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/Add*
T0	*#
_output_shapes
:’’’’’’’’’
¦
[dnn/input_from_feature_columns/input_layer/string_col_indicator/SparseToDense/default_valueConst*
_output_shapes
: *
dtype0	*
valueB	 R
’’’’’’’’’

Mdnn/input_from_feature_columns/input_layer/string_col_indicator/SparseToDenseSparseToDenseWdnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/indices[dnn/input_from_feature_columns/input_layer/string_col_indicator/to_sparse_input/dense_shapeZdnn/input_from_feature_columns/input_layer/string_col_indicator/hash_table_Lookup/SelectV2[dnn/input_from_feature_columns/input_layer/string_col_indicator/SparseToDense/default_value*
T0	*
Tindices0	*'
_output_shapes
:’’’’’’’’’

Mdnn/input_from_feature_columns/input_layer/string_col_indicator/one_hot/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

Odnn/input_from_feature_columns/input_layer/string_col_indicator/one_hot/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    

Mdnn/input_from_feature_columns/input_layer/string_col_indicator/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :

Pdnn/input_from_feature_columns/input_layer/string_col_indicator/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

Qdnn/input_from_feature_columns/input_layer/string_col_indicator/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
Ź
Gdnn/input_from_feature_columns/input_layer/string_col_indicator/one_hotOneHotMdnn/input_from_feature_columns/input_layer/string_col_indicator/SparseToDenseMdnn/input_from_feature_columns/input_layer/string_col_indicator/one_hot/depthPdnn/input_from_feature_columns/input_layer/string_col_indicator/one_hot/on_valueQdnn/input_from_feature_columns/input_layer/string_col_indicator/one_hot/off_value*
T0*+
_output_shapes
:’’’’’’’’’
Ø
Udnn/input_from_feature_columns/input_layer/string_col_indicator/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ž’’’’’’’’

Cdnn/input_from_feature_columns/input_layer/string_col_indicator/SumSumGdnn/input_from_feature_columns/input_layer/string_col_indicator/one_hotUdnn/input_from_feature_columns/input_layer/string_col_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:’’’’’’’’’
ø
Ednn/input_from_feature_columns/input_layer/string_col_indicator/ShapeShapeCdnn/input_from_feature_columns/input_layer/string_col_indicator/Sum*
T0*
_output_shapes
:

Sdnn/input_from_feature_columns/input_layer/string_col_indicator/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

Udnn/input_from_feature_columns/input_layer/string_col_indicator/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

Udnn/input_from_feature_columns/input_layer/string_col_indicator/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ķ
Mdnn/input_from_feature_columns/input_layer/string_col_indicator/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/string_col_indicator/ShapeSdnn/input_from_feature_columns/input_layer/string_col_indicator/strided_slice/stackUdnn/input_from_feature_columns/input_layer/string_col_indicator/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/string_col_indicator/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

Odnn/input_from_feature_columns/input_layer/string_col_indicator/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
£
Mdnn/input_from_feature_columns/input_layer/string_col_indicator/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/string_col_indicator/strided_sliceOdnn/input_from_feature_columns/input_layer/string_col_indicator/Reshape/shape/1*
N*
T0*
_output_shapes
:

Gdnn/input_from_feature_columns/input_layer/string_col_indicator/ReshapeReshapeCdnn/input_from_feature_columns/input_layer/string_col_indicator/SumMdnn/input_from_feature_columns/input_layer/string_col_indicator/Reshape/shape*
T0*'
_output_shapes
:’’’’’’’’’

6dnn/input_from_feature_columns/input_layer/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
Æ
1dnn/input_from_feature_columns/input_layer/concatConcatV2Fdnn/input_from_feature_columns/input_layer/array_col_indicator/Reshape5dnn/input_from_feature_columns/input_layer/f1/Reshape5dnn/input_from_feature_columns/input_layer/f2/ReshapeGdnn/input_from_feature_columns/input_layer/string_col_indicator/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
N*
T0*'
_output_shapes
:’’’’’’’’’
Å
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
:*
dtype0*
valueB"   C   
·
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *Ė¾
·
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *Ė>

Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:C*
dtype0

>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
¬
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:C

:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:C
Ī
dnn/hiddenlayer_0/kernel/part_0VarHandleOp*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: *
dtype0*
shape
:C*0
shared_name!dnn/hiddenlayer_0/kernel/part_0

@dnn/hiddenlayer_0/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
¤
&dnn/hiddenlayer_0/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*
dtype0

3dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:C*
dtype0
®
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
:C*
dtype0*
valueBC*    
Ä
dnn/hiddenlayer_0/bias/part_0VarHandleOp*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
_output_shapes
: *
dtype0*
shape:C*.
shared_namednn/hiddenlayer_0/bias/part_0

>dnn/hiddenlayer_0/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
: 

$dnn/hiddenlayer_0/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*
dtype0

1dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
:C*
dtype0

'dnn/hiddenlayer_0/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:C*
dtype0
v
dnn/hiddenlayer_0/kernelIdentity'dnn/hiddenlayer_0/kernel/ReadVariableOp*
T0*
_output_shapes

:C
”
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*'
_output_shapes
:’’’’’’’’’C

%dnn/hiddenlayer_0/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
:C*
dtype0
n
dnn/hiddenlayer_0/biasIdentity%dnn/hiddenlayer_0/bias/ReadVariableOp*
T0*
_output_shapes
:C

dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*'
_output_shapes
:’’’’’’’’’C
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’C
g
dnn/zero_fraction/SizeSizednn/hiddenlayer_0/Relu*
T0*
_output_shapes
: *
out_type0	
c
dnn/zero_fraction/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R’’’’

dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction/cond/SwitchSwitchdnn/zero_fraction/LessEqualdnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
m
dnn/zero_fraction/cond/switch_tIdentitydnn/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
k
dnn/zero_fraction/cond/switch_fIdentitydnn/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
h
dnn/zero_fraction/cond/pred_idIdentitydnn/zero_fraction/LessEqual*
T0
*
_output_shapes
: 

*dnn/zero_fraction/cond/count_nonzero/zerosConst ^dnn/zero_fraction/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
Ļ
-dnn/zero_fraction/cond/count_nonzero/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1*dnn/zero_fraction/cond/count_nonzero/zeros*
T0*'
_output_shapes
:’’’’’’’’’C
ę
4dnn/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu*:
_output_shapes(
&:’’’’’’’’’C:’’’’’’’’’C
”
)dnn/zero_fraction/cond/count_nonzero/CastCast-dnn/zero_fraction/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’C

*dnn/zero_fraction/cond/count_nonzero/ConstConst ^dnn/zero_fraction/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
±
2dnn/zero_fraction/cond/count_nonzero/nonzero_countSum)dnn/zero_fraction/cond/count_nonzero/Cast*dnn/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 

dnn/zero_fraction/cond/CastCast2dnn/zero_fraction/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 

,dnn/zero_fraction/cond/count_nonzero_1/zerosConst ^dnn/zero_fraction/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
Ó
/dnn/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch,dnn/zero_fraction/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:’’’’’’’’’C
č
6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu*:
_output_shapes(
&:’’’’’’’’’C:’’’’’’’’’C
„
+dnn/zero_fraction/cond/count_nonzero_1/CastCast/dnn/zero_fraction/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*'
_output_shapes
:’’’’’’’’’C

,dnn/zero_fraction/cond/count_nonzero_1/ConstConst ^dnn/zero_fraction/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
·
4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countSum+dnn/zero_fraction/cond/count_nonzero_1/Cast,dnn/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
¤
dnn/zero_fraction/cond/MergeMerge4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countdnn/zero_fraction/cond/Cast*
N*
T0	*
_output_shapes
: : 

(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 

)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 
{
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*

DstT0*

SrcT0	*
_output_shapes
: 
°
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
 
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values
Æ
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
T0*
_output_shapes
: 

$dnn/dnn/hiddenlayer_0/activation/tagConst*
_output_shapes
: *
dtype0*1
value(B& B dnn/dnn/hiddenlayer_0/activation

 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: 
·
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
:*
dtype0*
valueB"C      
©
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *:Ķ¾
©
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
dtype0*
valueB
 *:Ķ>
š
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:C*
dtype0
ž
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 

7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:C

3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:C
¹
dnn/logits/kernel/part_0VarHandleOp*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: *
dtype0*
shape
:C*)
shared_namednn/logits/kernel/part_0

9dnn/logits/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel/part_0*
_output_shapes
: 

dnn/logits/kernel/part_0/AssignAssignVariableOpdnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*
dtype0

,dnn/logits/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
_output_shapes

:C*
dtype0
 
(dnn/logits/bias/part_0/Initializer/zerosConst*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
:*
dtype0*
valueB*    
Æ
dnn/logits/bias/part_0VarHandleOp*)
_class
loc:@dnn/logits/bias/part_0*
_output_shapes
: *
dtype0*
shape:*'
shared_namednn/logits/bias/part_0
}
7dnn/logits/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias/part_0*
_output_shapes
: 

dnn/logits/bias/part_0/AssignAssignVariableOpdnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*
dtype0
}
*dnn/logits/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
_output_shapes
:*
dtype0
y
 dnn/logits/kernel/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
_output_shapes

:C*
dtype0
h
dnn/logits/kernelIdentity dnn/logits/kernel/ReadVariableOp*
T0*
_output_shapes

:C
x
dnn/logits/MatMulMatMuldnn/hiddenlayer_0/Reludnn/logits/kernel*
T0*'
_output_shapes
:’’’’’’’’’
q
dnn/logits/bias/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
_output_shapes
:*
dtype0
`
dnn/logits/biasIdentitydnn/logits/bias/ReadVariableOp*
T0*
_output_shapes
:
s
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*'
_output_shapes
:’’’’’’’’’
e
dnn/zero_fraction_1/SizeSizednn/logits/BiasAdd*
T0*
_output_shapes
: *
out_type0	
e
dnn/zero_fraction_1/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R’’’’

dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 

dnn/zero_fraction_1/cond/SwitchSwitchdnn/zero_fraction_1/LessEqualdnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_1/cond/switch_tIdentity!dnn/zero_fraction_1/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_1/cond/switch_fIdentitydnn/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_1/cond/pred_idIdentitydnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: 

,dnn/zero_fraction_1/cond/count_nonzero/zerosConst"^dnn/zero_fraction_1/cond/switch_t*
_output_shapes
: *
dtype0*
valueB
 *    
Õ
/dnn/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_1/cond/count_nonzero/zeros*
T0*'
_output_shapes
:’’’’’’’’’
ā
6dnn/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_1/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:’’’’’’’’’:’’’’’’’’’
„
+dnn/zero_fraction_1/cond/count_nonzero/CastCast/dnn/zero_fraction_1/cond/count_nonzero/NotEqual*

DstT0*

SrcT0
*'
_output_shapes
:’’’’’’’’’
”
,dnn/zero_fraction_1/cond/count_nonzero/ConstConst"^dnn/zero_fraction_1/cond/switch_t*
_output_shapes
:*
dtype0*
valueB"       
·
4dnn/zero_fraction_1/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_1/cond/count_nonzero/Cast,dnn/zero_fraction_1/cond/count_nonzero/Const*
T0*
_output_shapes
: 

dnn/zero_fraction_1/cond/CastCast4dnn/zero_fraction_1/cond/count_nonzero/nonzero_count*

DstT0	*

SrcT0*
_output_shapes
: 

.dnn/zero_fraction_1/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_1/cond/switch_f*
_output_shapes
: *
dtype0*
valueB
 *    
Ł
1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_1/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:’’’’’’’’’
ä
8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_1/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:’’’’’’’’’:’’’’’’’’’
©
-dnn/zero_fraction_1/cond/count_nonzero_1/CastCast1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual*

DstT0	*

SrcT0
*'
_output_shapes
:’’’’’’’’’
£
.dnn/zero_fraction_1/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_1/cond/switch_f*
_output_shapes
:*
dtype0*
valueB"       
½
6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_1/cond/count_nonzero_1/Cast.dnn/zero_fraction_1/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
Ŗ
dnn/zero_fraction_1/cond/MergeMerge6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_1/cond/Cast*
N*
T0	*
_output_shapes
: : 

*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Sizednn/zero_fraction_1/cond/Merge*
T0	*
_output_shapes
: 

+dnn/zero_fraction_1/counts_to_fraction/CastCast*dnn/zero_fraction_1/counts_to_fraction/sub*

DstT0*

SrcT0	*
_output_shapes
: 

-dnn/zero_fraction_1/counts_to_fraction/Cast_1Castdnn/zero_fraction_1/Size*

DstT0*

SrcT0	*
_output_shapes
: 
¶
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 

+dnn/dnn/logits/fraction_of_zero_values/tagsConst*
_output_shapes
: *
dtype0*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values
£
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst*
_output_shapes
: *
dtype0**
value!B Bdnn/dnn/logits/activation
x
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
W
dnn/head/logits/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
k
)dnn/head/logits/assert_rank_at_least/rankConst*
_output_shapes
: *
dtype0*
value	B :
[
Sdnn/head/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
L
Ddnn/head/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
\
dnn/head/predictions/ShapeShapednn/logits/BiasAdd*
T0*
_output_shapes
:
r
(dnn/head/predictions/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
t
*dnn/head/predictions/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
t
*dnn/head/predictions/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

"dnn/head/predictions/strided_sliceStridedSlicednn/head/predictions/Shape(dnn/head/predictions/strided_slice/stack*dnn/head/predictions/strided_slice/stack_1*dnn/head/predictions/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
b
 dnn/head/predictions/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
b
 dnn/head/predictions/range/limitConst*
_output_shapes
: *
dtype0*
value	B :
b
 dnn/head/predictions/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
„
dnn/head/predictions/rangeRange dnn/head/predictions/range/start dnn/head/predictions/range/limit dnn/head/predictions/range/delta*
_output_shapes
:
e
#dnn/head/predictions/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 

dnn/head/predictions/ExpandDims
ExpandDimsdnn/head/predictions/range#dnn/head/predictions/ExpandDims/dim*
T0*
_output_shapes

:
g
%dnn/head/predictions/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
¤
#dnn/head/predictions/Tile/multiplesPack"dnn/head/predictions/strided_slice%dnn/head/predictions/Tile/multiples/1*
N*
T0*
_output_shapes
:

dnn/head/predictions/TileTilednn/head/predictions/ExpandDims#dnn/head/predictions/Tile/multiples*
T0*'
_output_shapes
:’’’’’’’’’
^
dnn/head/predictions/Shape_1Shapednn/logits/BiasAdd*
T0*
_output_shapes
:
t
*dnn/head/predictions/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
v
,dnn/head/predictions/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
v
,dnn/head/predictions/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
 
$dnn/head/predictions/strided_slice_1StridedSlicednn/head/predictions/Shape_1*dnn/head/predictions/strided_slice_1/stack,dnn/head/predictions/strided_slice_1/stack_1,dnn/head/predictions/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

'dnn/head/predictions/ExpandDims_1/inputConst*
_output_shapes
:*
dtype0*,
value#B!BeeeBdddBcccBbbbBaaa
g
%dnn/head/predictions/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Ø
!dnn/head/predictions/ExpandDims_1
ExpandDims'dnn/head/predictions/ExpandDims_1/input%dnn/head/predictions/ExpandDims_1/dim*
T0*
_output_shapes

:
i
'dnn/head/predictions/Tile_1/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :
Ŗ
%dnn/head/predictions/Tile_1/multiplesPack$dnn/head/predictions/strided_slice_1'dnn/head/predictions/Tile_1/multiples/1*
N*
T0*
_output_shapes
:

dnn/head/predictions/Tile_1Tile!dnn/head/predictions/ExpandDims_1%dnn/head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:’’’’’’’’’
s
(dnn/head/predictions/class_ids/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’

dnn/head/predictions/class_idsArgMaxdnn/logits/BiasAdd(dnn/head/predictions/class_ids/dimension*
T0*#
_output_shapes
:’’’’’’’’’
p
%dnn/head/predictions/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
’’’’’’’’’
Ø
!dnn/head/predictions/ExpandDims_2
ExpandDimsdnn/head/predictions/class_ids%dnn/head/predictions/ExpandDims_2/dim*
T0	*'
_output_shapes
:’’’’’’’’’

.dnn/head/predictions/class_string_lookup/ConstConst*
_output_shapes
:*
dtype0*,
value#B!BeeeBdddBcccBbbbBaaa
o
-dnn/head/predictions/class_string_lookup/SizeConst*
_output_shapes
: *
dtype0*
value	B :
v
4dnn/head/predictions/class_string_lookup/range/startConst*
_output_shapes
: *
dtype0*
value	B : 
v
4dnn/head/predictions/class_string_lookup/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :
ī
.dnn/head/predictions/class_string_lookup/rangeRange4dnn/head/predictions/class_string_lookup/range/start-dnn/head/predictions/class_string_lookup/Size4dnn/head/predictions/class_string_lookup/range/delta*
_output_shapes
:

-dnn/head/predictions/class_string_lookup/CastCast.dnn/head/predictions/class_string_lookup/range*

DstT0	*

SrcT0*
_output_shapes
:
t
0dnn/head/predictions/class_string_lookup/Const_1Const*
_output_shapes
: *
dtype0*
valueB	 BUNK
Ą
3dnn/head/predictions/class_string_lookup/hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0	*@
shared_name1/hash_table_e7895b38-f856-4c60-a05d-bf9b213a4032*
value_dtype0

Gdnn/head/predictions/class_string_lookup/table_init/LookupTableImportV2LookupTableImportV23dnn/head/predictions/class_string_lookup/hash_table-dnn/head/predictions/class_string_lookup/Cast.dnn/head/predictions/class_string_lookup/Const*	
Tin0	*

Tout0

8dnn/head/predictions/hash_table_Lookup/LookupTableFindV2LookupTableFindV23dnn/head/predictions/class_string_lookup/hash_table!dnn/head/predictions/ExpandDims_20dnn/head/predictions/class_string_lookup/Const_1*	
Tin0	*

Tout0*'
_output_shapes
:’’’’’’’’’
s
"dnn/head/predictions/probabilitiesSoftmaxdnn/logits/BiasAdd*
T0*'
_output_shapes
:’’’’’’’’’
`
dnn/head/ShapeShape"dnn/head/predictions/probabilities*
T0*
_output_shapes
:
f
dnn/head/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
h
dnn/head/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
h
dnn/head/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Ś
dnn/head/strided_sliceStridedSlicednn/head/Shapednn/head/strided_slice/stackdnn/head/strided_slice/stack_1dnn/head/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
y
dnn/head/ExpandDims/inputConst*
_output_shapes
:*
dtype0*,
value#B!BeeeBdddBcccBbbbBaaa
Y
dnn/head/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
~
dnn/head/ExpandDims
ExpandDimsdnn/head/ExpandDims/inputdnn/head/ExpandDims/dim*
T0*
_output_shapes

:
[
dnn/head/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :

dnn/head/Tile/multiplesPackdnn/head/strided_slicednn/head/Tile/multiples/1*
N*
T0*
_output_shapes
:
u
dnn/head/TileTilednn/head/ExpandDimsdnn/head/Tile/multiples*
T0*'
_output_shapes
:’’’’’’’’’

initNoOp
ń
init_all_tablesNoOpH^dnn/head/predictions/class_string_lookup/table_init/LookupTableImportV2^dnn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/table_init/InitializeTableFromTextFileV2^dnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/table_init/InitializeTableFromTextFileV2

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
r
save/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
:C*
dtype0
X
save/IdentityIdentitysave/Read/ReadVariableOp*
T0*
_output_shapes
:C
^
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes
:C
z
save/Read_1/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:C*
dtype0
`
save/Identity_2Identitysave/Read_1/ReadVariableOp*
T0*
_output_shapes

:C
d
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes

:C
m
save/Read_2/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
_output_shapes
:*
dtype0
\
save/Identity_4Identitysave/Read_2/ReadVariableOp*
T0*
_output_shapes
:
`
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
T0*
_output_shapes
:
s
save/Read_3/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
_output_shapes

:C*
dtype0
`
save/Identity_6Identitysave/Read_3/ReadVariableOp*
T0*
_output_shapes

:C
d
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
T0*
_output_shapes

:C

save/StringJoin/inputs_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_e3d52e8a93854bfdbe4c4f4a805cd780/part
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
{
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBBglobal_step
t
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step"/device:CPU:0*
dtypes
2	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :

save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 

save/Read_4/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0"/device:CPU:0*
_output_shapes
:C*
dtype0
k
save/Identity_8Identitysave/Read_4/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:C
`
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
T0*
_output_shapes
:C

save/Read_5/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0"/device:CPU:0*
_output_shapes

:C*
dtype0
p
save/Identity_10Identitysave/Read_5/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:C
f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:C
|
save/Read_6/ReadVariableOpReadVariableOpdnn/logits/bias/part_0"/device:CPU:0*
_output_shapes
:*
dtype0
l
save/Identity_12Identitysave/Read_6/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
T0*
_output_shapes
:

save/Read_7/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0"/device:CPU:0*
_output_shapes

:C*
dtype0
p
save/Identity_14Identitysave/Read_7/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:C
f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
T0*
_output_shapes

:C
Ę
save/SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*i
value`B^Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/logits/biasBdnn/logits/kernel
¤
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B67 0,67B18 67 0,18:0,67B5 0,5B67 5 0,67:0,5
Õ
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_9save/Identity_11save/Identity_13save/Identity_15"/device:CPU:0*
dtypes
2
Ø
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
Ō
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
Ø
save/Identity_16Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
~
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0* 
valueBBglobal_step
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2	
s
save/AssignAssignglobal_stepsave/RestoreV2*
T0	*
_class
loc:@global_step*
_output_shapes
: 
(
save/restore_shardNoOp^save/Assign
É
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*i
value`B^Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/logits/biasBdnn/logits/kernel
§
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B67 0,67B18 67 0,18:0,67B5 0,5B67 5 0,67:0,5
Ä
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*4
_output_shapes"
 :C:C::C*
dtypes
2
b
save/Identity_17Identitysave/RestoreV2_1"/device:CPU:0*
T0*
_output_shapes
:C
v
save/AssignVariableOpAssignVariableOpdnn/hiddenlayer_0/bias/part_0save/Identity_17"/device:CPU:0*
dtype0
h
save/Identity_18Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes

:C
z
save/AssignVariableOp_1AssignVariableOpdnn/hiddenlayer_0/kernel/part_0save/Identity_18"/device:CPU:0*
dtype0
d
save/Identity_19Identitysave/RestoreV2_1:2"/device:CPU:0*
T0*
_output_shapes
:
q
save/AssignVariableOp_2AssignVariableOpdnn/logits/bias/part_0save/Identity_19"/device:CPU:0*
dtype0
h
save/Identity_20Identitysave/RestoreV2_1:3"/device:CPU:0*
T0*
_output_shapes

:C
s
save/AssignVariableOp_3AssignVariableOpdnn/logits/kernel/part_0save/Identity_20"/device:CPU:0*
dtype0

save/restore_shard_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_2^save/AssignVariableOp_3"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"?
save/Const:0save/Identity_16:0save/restore_all (5 @F8"ė
asset_filepaths×
Ō
gdnn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/asset_path:0
idnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/asset_path:0"Ų
cond_contextĒÄ
¬
 dnn/zero_fraction/cond/cond_text dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_t:0 *Ą
dnn/hiddenlayer_0/Relu:0
dnn/zero_fraction/cond/Cast:0
+dnn/zero_fraction/cond/count_nonzero/Cast:0
,dnn/zero_fraction/cond/count_nonzero/Const:0
6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
/dnn/zero_fraction/cond/count_nonzero/NotEqual:0
4dnn/zero_fraction/cond/count_nonzero/nonzero_count:0
,dnn/zero_fraction/cond/count_nonzero/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_t:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0R
dnn/hiddenlayer_0/Relu:06dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1

"dnn/zero_fraction/cond/cond_text_1 dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_f:0*Æ
dnn/hiddenlayer_0/Relu:0
-dnn/zero_fraction/cond/count_nonzero_1/Cast:0
.dnn/zero_fraction/cond/count_nonzero_1/Const:0
8dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
1dnn/zero_fraction/cond/count_nonzero_1/NotEqual:0
6dnn/zero_fraction/cond/count_nonzero_1/nonzero_count:0
.dnn/zero_fraction/cond/count_nonzero_1/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_f:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0T
dnn/hiddenlayer_0/Relu:08dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Ā
"dnn/zero_fraction_1/cond/cond_text"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_t:0 *Š
dnn/logits/BiasAdd:0
dnn/zero_fraction_1/cond/Cast:0
-dnn/zero_fraction_1/cond/count_nonzero/Cast:0
.dnn/zero_fraction_1/cond/count_nonzero/Const:0
8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_1/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_1/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_1/cond/count_nonzero/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_t:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0P
dnn/logits/BiasAdd:08dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
Æ
$dnn/zero_fraction_1/cond/cond_text_1"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_f:0*½
dnn/logits/BiasAdd:0
/dnn/zero_fraction_1/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_1/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_1/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_f:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0R
dnn/logits/BiasAdd:0:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"õ
saved_model_assetsŽ*Ū
©
+type.googleapis.com/tensorflow.AssetFileDefz
i
gdnn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/asset_path:0array_col.csv
¬
+type.googleapis.com/tensorflow.AssetFileDef}
k
idnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/asset_path:0string_col.csv"%
saved_model_main_op


group_deps"­
	summaries

/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0"š
table_initializerŚ
×
dnn/input_from_feature_columns/input_layer/array_col_indicator/array_col_lookup/hash_table/table_init/InitializeTableFromTextFileV2
dnn/input_from_feature_columns/input_layer/string_col_indicator/string_col_lookup/hash_table/table_init/InitializeTableFromTextFileV2
Gdnn/head/predictions/class_string_lookup/table_init/LookupTableImportV2"å
trainable_variablesĶŹ
ģ
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernelC  "C(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
Ö
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/biasC "C(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
É
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernelC  "C(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
³
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08"·
	variables©¦
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
ģ
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernelC  "C(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
Ö
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/biasC "C(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
É
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernelC  "C(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
³
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08* 
predict
<
	array_col/
Placeholder_3:0’’’’’’’’’’’’’’’’’’
&
f1 
Placeholder:0’’’’’’’’’
(
f2"
Placeholder_1:0’’’’’’’’’
0

string_col"
Placeholder_2:0’’’’’’’’’C
all_class_ids2
dnn/head/predictions/Tile:0’’’’’’’’’C
all_classes4
dnn/head/predictions/Tile_1:0’’’’’’’’’G
	class_ids:
#dnn/head/predictions/ExpandDims_2:0	’’’’’’’’’\
classesQ
:dnn/head/predictions/hash_table_Lookup/LookupTableFindV2:0’’’’’’’’’5
logits+
dnn/logits/BiasAdd:0’’’’’’’’’L
probabilities;
$dnn/head/predictions/probabilities:0’’’’’’’’’tensorflow/serving/predict*Ø
serving_default
<
	array_col/
Placeholder_3:0’’’’’’’’’’’’’’’’’’
&
f1 
Placeholder:0’’’’’’’’’
(
f2"
Placeholder_1:0’’’’’’’’’
0

string_col"
Placeholder_2:0’’’’’’’’’C
all_class_ids2
dnn/head/predictions/Tile:0’’’’’’’’’C
all_classes4
dnn/head/predictions/Tile_1:0’’’’’’’’’G
	class_ids:
#dnn/head/predictions/ExpandDims_2:0	’’’’’’’’’\
classesQ
:dnn/head/predictions/hash_table_Lookup/LookupTableFindV2:0’’’’’’’’’5
logits+
dnn/logits/BiasAdd:0’’’’’’’’’L
probabilities;
$dnn/head/predictions/probabilities:0’’’’’’’’’tensorflow/serving/predict