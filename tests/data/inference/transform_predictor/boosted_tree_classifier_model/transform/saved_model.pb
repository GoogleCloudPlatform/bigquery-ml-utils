đR
č
Ë

D
AddV2
x"T
y"T
z"T"
Ttype:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
ł
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
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
Á
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
executor_typestring ¨
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
	separatorstring "serve*2.12.02unknown8Í9
m
serving_default_f1Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
m
serving_default_f2Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
m
serving_default_f3Placeholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

serving_default_f4Placeholder*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

PartitionedCallPartitionedCallserving_default_f1serving_default_f2serving_default_f3serving_default_f4*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_signature_wrapper_43

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*B
value9B7 B1


signatures* 

serving_default* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
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
GPU 2J 8 *$
fR
__inference__traced_save_72

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
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
GPU 2J 8 *'
f"R 
__inference__traced_restore_82Ĺ,
Ú
w
__inference_<lambda>_27
f2
f3
f4
f1
identity

identity_1

identity_2

identity_3T
model/tf.math.add/AddAddV2f1f2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙S
IdentityIdentityf4*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙H

Identity_1Identityf1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙_

Identity_2Identitymodel/tf.math.add/Add:z:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙H

Identity_3Identityf3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namef2:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namef3:TP
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namef4:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namef1
í
D
__inference__traced_restore_82
file_prefix

identity_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B Ł
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
2Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


 __inference_signature_wrapper_43
f1
f2
f3
f4
identity

identity_1

identity_2

identity_3Ć
PartitionedCallPartitionedCallf2f3f4f1*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 * 
fR
__inference_<lambda>_27i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙^

Identity_1IdentityPartitionedCall:output:1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^

Identity_2IdentityPartitionedCall:output:2*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙^

Identity_3IdentityPartitionedCall:output:3*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:G C
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namef1:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namef2:GC
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namef3:TP
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

_user_specified_namef4
Ú
h
__inference__traced_save_72
file_prefix
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B Ř
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: "ˇ
J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ĺ
serving_defaultą
-
f1'
serving_default_f1:0˙˙˙˙˙˙˙˙˙
-
f2'
serving_default_f2:0˙˙˙˙˙˙˙˙˙
-
f3'
serving_default_f3:0˙˙˙˙˙˙˙˙˙
:
f44
serving_default_f4:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙>
	array_col1
PartitionedCall:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
f1$
PartitionedCall:1˙˙˙˙˙˙˙˙˙*
f2$
PartitionedCall:2˙˙˙˙˙˙˙˙˙2

string_col$
PartitionedCall:3˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:˝	
ç

signaturesBÄ
__inference_<lambda>_27f2f3f4f1"
˛
FullArgSpec
args 
varargs
jfeatures
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z
signatures
,
serving_default"
signature_map
ĚBÉ
 __inference_signature_wrapper_43f1f2f3f4"
˛
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
annotationsŞ *
 Ů
__inference_<lambda>_27˝˘
}˘z
xu

f2˙˙˙˙˙˙˙˙˙

f3˙˙˙˙˙˙˙˙˙
%"
f4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

f1˙˙˙˙˙˙˙˙˙
Ş "ŻŞŤ
9
	array_col,)
	array_col˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

f1
f1˙˙˙˙˙˙˙˙˙

f2
f2˙˙˙˙˙˙˙˙˙
.

string_col 

string_col˙˙˙˙˙˙˙˙˙÷
 __inference_signature_wrapper_43Ň˘
˘ 
Ş

f1
f1˙˙˙˙˙˙˙˙˙

f2
f2˙˙˙˙˙˙˙˙˙

f3
f3˙˙˙˙˙˙˙˙˙
+
f4%"
f4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"ŻŞŤ
9
	array_col,)
	array_col˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

f1
f1˙˙˙˙˙˙˙˙˙

f2
f2˙˙˙˙˙˙˙˙˙
.

string_col 

string_col˙˙˙˙˙˙˙˙˙