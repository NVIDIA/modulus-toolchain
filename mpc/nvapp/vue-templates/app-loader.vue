<template>
    <div v-resize="wresize">
        <v-app-bar dense dark color="green">
            <!-- <v-img contain src="/files/resources/nvidia-logo.png" max-width="150"></v-img> -->
            <v-toolbar-title>App Loader</v-toolbar-title>
            <v-spacer></v-spacer>
            <v-dialog v-model="app_dialog" width="500">
                <template v-slot:activator="{ on, attrs }">
                    <v-btn dark v-bind="attrs" v-on="on">
                        Change App
                    </v-btn>
                </template>
                <div style="z-index:99">
                    <v-card>
                        <v-card-title class="text-h5 grey lighten-2">
                            Select App
                        </v-card-title>
                        <v-row class="pa-2" align="center" no-gutters>
                            <v-col cols=4>
                                <v-select label="Select App" v-model="selected_app" :items="apps"></v-select>
                            </v-col>
                            <v-col cols=4>
                                <v-btn small :loading="loading_app" @click="reload" :color="loading_app_result">
                                    <v-icon small>mdi-refresh</v-icon>
                                </v-btn>
                            </v-col>
                        </v-row>
                    </v-card>
                </div>
            </v-dialog>
            <v-row align="center">
                <v-col cols=4>
                    <v-select label="Select App" v-model="selected_app" :items="apps"></v-select>
                </v-col>
                <v-col cols=4>
                    <v-row>
                        <v-col cols=auto>
                            <v-btn small :loading="loading_app" @click="reload" :color="loading_app_result">
                                <v-icon small>mdi-refresh</v-icon>
                            </v-btn>
                        </v-col>
                        <v-col cols=auto>
                            <v-dialog v-model="output_dialog" max-width="1000">
                                <template v-slot:activator="{ on, attrs }">
                                    <v-btn small v-bind="attrs" v-on="on">
                                        Output
                                    </v-btn>
                                </template>
                                <div style="z-index:99">
                                    <v-card>
                                        <v-card-title class="text-h5 grey lighten-2">
                                            App Text Output (Errors)
                                            <v-spacer></v-spacer>
                                            <v-btn small outlined @click="clear_output">clear</v-btn>
                                        </v-card-title>
                                        <v-row class="pa-2" align="center" no-gutters>
                                            <v-col>
                                                <app-output></app-output>
                                            </v-col>

                                        </v-row>
                                    </v-card>
                                </div>
                            </v-dialog>
                        </v-col>
                    </v-row>
                </v-col>


            </v-row>

        </v-app-bar>

        <v-content style="z-index:-1">
            <v-alert v-if="info" type="error">
                <kbd>{{info}}</kbd>
                <v-btn @click="output_dialog=true">show output</v-btn>
            </v-alert>
            <div v-show="!output_dialog">
                <mycontent />
            </div>
        </v-content>

        <!-- <v-footer dense app>
            Footer
        </v-footer> -->
    </div>
</template>

<script>
    module.exports = {
        methods: {
            wresize() {
                this.wsz = { x: window.innerWidth, y: window.innerHeight }
            }
        }
    }
</script>