<template>
    <div>
        <v-app-bar dense dark color="darkgreen">
            <v-toolbar-title>Modulus Project Configurator (mpc)
            </v-toolbar-title>
            <v-row align="center" dense>


                <v-col cols="auto" align="right">
                    <v-dialog v-model="problem_dialog" max-width="1000">
                        <template v-slot:activator="{ on, attrs }">
                            <v-btn small v-bind="attrs" v-on="on" @click="load_problem">
                                problem
                            </v-btn>
                        </template>
                        <div style="z-index:99">
                            <v-card>
                                <v-card-title class="text-h5 grey lighten-2">
                                    Problem Summary
                                    <v-spacer></v-spacer><kbd>mtc show problem</kbd>
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
                    <config></config>
                </v-col>

            </v-row>
        </div>


    </div>

</template>