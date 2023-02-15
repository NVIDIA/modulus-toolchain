<template>

    <v-container>
        <v-card>
            <v-container>
                <v-row>
                    <v-col cols="auto">Name:</v-col>
                    <v-col cols="auto"> <kbd>{{nn_name}}</kbd></v-col>
                    <v-col>
                        <!-- <v-btn rounded small outlined>{{nn_data['nn_type']}}</v-btn> -->
                        <v-menu offset-y>
                            <template v-slot:activator="{ on, attrs }">
                                <!-- <v-btn color="primary" dark v-bind="attrs" v-on="on">
                                    Dropdown
                                </v-btn> -->
                                <v-btn v-bind="attrs" v-on="on" rounded small outlined>{{nn_data['nn_type']}}</v-btn>
                            </template>
                            <v-list>
                                <v-list-item v-for="(item, index) in nn_types" :key="index">
                                    <v-list-item-title @click="switch_nn_type(item)">{{ item }}</v-list-item-title>
                                </v-list-item>
                            </v-list>
                        </v-menu>
                    </v-col>
                </v-row>
                <v-row v-for="(v,k) in nn_schema" :key="k">
                    <!-- {{k}}:{{v}} -->
                    <v-text-field number v-if="v.type==='int'" hint="number" :label="k" v-model="nn_data[k]">
                    </v-text-field>
                    <v-text-field number v-if="v.type==='string_tuple'" hint="number" :label="k" v-model="nn_data[k]">
                    </v-text-field>
                    <v-btn small outlined v-if="v.type==='bool'" :color="nn_data[k] ? 'green':'red'"
                        @click="nn_data[k] = !nn_data[k]">{{k}}:<v-icon dark right>
                            {{nn_data[k] ? "mdi-checkbox-marked-circle" : "mdi-cancel"}}
                        </v-icon>
                    </v-btn>
                </v-row>

                <v-row>
                    <v-btn @click='check'>check</v-btn>
                </v-row>
            </v-container>

            <!-- <v-card-body>there</v-card-body> -->
        </v-card>
    </v-container>
</template>