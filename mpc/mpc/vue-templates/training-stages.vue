<template>
    <v-card class="pa-2">
        <v-row no-gutters>

            <v-row v-if="editing_metadata">
                <v-col>
                    <project-metadata></project-metadata>
                </v-col>
                <v-col cols="auto">
                    <v-icon color="green" @click="save_metadata(); ">mdi-check</v-icon>
                </v-col>
                <v-col cols="auto">
                    <v-icon color="red" @click="editing_metadata=!editing_metadata ">mdi-cancel</v-icon>
                </v-col>
            </v-row>
            <v-row v-if="!editing_metadata" no-gutters>
                <v-col>
                    <v-row align="center">
                        <!-- <v-col cols="auto">
                            <div class="overline">PROJECT</div>
                        </v-col> -->
                        <v-col cols="auto">
                            <!-- <h2>{{metadata['project-name']}}</h2> -->
                            <v-card-title>{{metadata['project-name']}}</v-card-title>

                        </v-col>
                        <v-col>

                            <v-tooltip bottom>
                                <template v-slot:activator="{ on, attrs }">
                                    <v-icon v-bind="attrs" v-on="on">mdi-information</v-icon>

                                </template><span>
                                    <pre>{{metadata['project-description']}}</pre>
                                </span>
                            </v-tooltip>
                        </v-col>
                        <v-col cols="auto">
                            <v-icon @click="editing_metadata=!editing_metadata">mdi-pencil-outline</v-icon>
                        </v-col>
                    </v-row>
                    <!-- <v-row>
                        <v-col>
                            <div class="overline">description</div>
                        </v-col>
                    </v-row> -->
                    <!-- <v-row>
                        <v-col>
                            <div class="body-2">{{metadata['project-description']}}</div>
                        </v-col>
                    </v-row> -->
                </v-col>
            </v-row>

        </v-row>
        <v-row>
            <v-col>
                <v-divider></v-divider>
            </v-col>
        </v-row>
        <v-row class="pa-1">
            <v-col>
                <v-btn small @click="add_new_stage">new training branch</v-btn>
            </v-col>
            <v-col>

                <v-tooltip bottom>
                    <template v-slot:activator="{ on, attrs }">

                        <v-chip v-bind="attrs" v-on="on" small @click="show_stage_dag=!show_stage_dag">
                            <v-icon left small>{{show_stage_dag ? 'mdi-eye' : 'mdi-eye-off'}}
                            </v-icon>
                            Stage DAG
                        </v-chip>
                    </template><span>
                        {{show_stage_dag ? 'Hide':'Show'}} stage dependency graph
                    </span>
                </v-tooltip>


            </v-col>
            <v-col>
                <v-tooltip bottom>
                    <template v-slot:activator="{ on, attrs }">

                        <v-chip v-bind="attrs" v-on="on" small @click="show_stage_ui=!show_stage_ui">
                            <v-icon left small>{{show_stage_ui ? 'mdi-eye' : 'mdi-eye-off'}}
                            </v-icon>
                            Stage Config
                        </v-chip>
                    </template><span>
                        {{show_stage_ui ? 'Hide':'Show'}} stage configuration UI
                    </span>
                </v-tooltip>


            </v-col>
            <v-col cols="auto">
                <v-btn small :disabled="!not_saved" @click="save_config_to_file">save</v-btn>
            </v-col>

        </v-row>

        <v-row>
            <v-col>
                <v-chip class="ma-1" small v-for="stage in stages" @click="select_stage(stage)">{{stage}}</v-chip>
            </v-col>
        </v-row>
        <v-row>
            <v-col cols=6 v-if="show_stage_dag">

                <div v-html=svgstr style="overflow: auto"></div>

            </v-col>

        </v-row>
        <v-row>
            <v-col v-if="show_stage_ui">
                <selected-stage></selected-stage>
            </v-col>
        </v-row>
        <!-- <selected-stage></selected-stage> -->
    </v-card>
</template>