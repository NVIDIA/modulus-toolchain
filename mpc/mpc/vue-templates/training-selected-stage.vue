<template>
    <v-card class="pa-1">
        <v-container>
            <h2>Configure: {{stage_id}}</h2>
            <v-row>
                <v-col>
                    <v-btn small @click="update">update</v-btn>
                </v-col>
                <v-col cols="auto">
                    <v-btn small @click="add_child_stage">+ child stage</v-btn>
                </v-col>
                <v-col cols="auto">
                    <v-tooltip bottom>
                        <template v-slot:activator="{ on, attrs }">
                            <v-btn v-bind="attrs" v-on="on" small @click="add_child_stage" color="red">remove</v-btn>
                        </template><span>Tooltip</span>
                    </v-tooltip>
                </v-col>
            </v-row>
            <v-row>
                <v-textarea outlined rows=2 label="Stage Description" v-model=stage.description></v-textarea>
            </v-row>
            <v-row>
                <config-stage></config-stage>
            </v-row>
            <!--  -->
        </v-container>
    </v-card>
</template>