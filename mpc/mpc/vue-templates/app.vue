<template>
    <div>
        <v-app-bar dense dark color="darkgreen">
            <v-toolbar-title>Modulus Project Configurator (mpc)
            </v-toolbar-title>
            <v-row align="center" dense>

                <v-col align="right">
                    <div class="">
                        PROJECT
                    </div>
                </v-col>
                <v-col cols="auto">
                    <kbd>{{projname}}</kbd>
                </v-col>
                <v-col cols="auto" align="right">
                    <v-dialog v-model="problem_dialog" max-width="1000">
                        <template v-slot:activator="{ on, attrs }">
                            <v-btn small v-bind="attrs" v-on="on" @click="load_problem">
                                problem
                            </v-btn>
                            <v-btn small v-bind="attrs" v-on="on" @click="load_problem_1storder">
                                problem 1st-order
                            </v-btn>
                            <v-btn small v-bind="attrs" v-on="on" @click="load_show_training">
                                training
                            </v-btn>
                        </template>
                        <div style="z-index:99">
                            <v-card>
                                <v-card-title class="text-h5 grey lighten-2">
                                    {{summary_type}} Summary
                                    <v-spacer></v-spacer><kbd>{{problem_show_cmd}}</kbd>
                                    <!-- <v-btn small outlined @click="clear_output">clear</v-btn> -->
                                </v-card-title>
                                <v-row class="pa-2" align="center" no-gutters>
                                    <v-col>
                                        <kbd>{{problemstr}}</kbd>
                                    </v-col>

                                </v-row>
                            </v-card>
                        </div>
                    </v-dialog>
                </v-col>
            </v-row>
        </v-app-bar>

        <div class="pa-1">
            <v-row>
                <v-col>

                    <training></training>

                </v-col>
            </v-row>
            <v-row>
                <v-col>
                    <training-plots></training-plots>
                </v-col>
            </v-row>
        </div>


    </div>

</template>